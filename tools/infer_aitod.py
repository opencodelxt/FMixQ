import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

from src.core import YAMLConfig

"""
AITOD 推理与可视化脚本（与 GT 可视化一致的绘制风格）

用法示例：
python tools/infer_aitod.py \
  -c /path/to/config.yml \
  -r /path/to/checkpoint.pth \
  --img-folder /path/to/images \
  --output-dir ./pred_vis_aitod \
  --score-thresh 0.5 \
  --input-size 640 \
  -d cuda:0
"""

# AITOD 类别（按连续索引 0..7 对应）
AITOD_CLASSES = [
    'airplane',
    'bridge',
    'storage-tank',
    'ship',
    'swimming-pool',
    'vehicle',
    'person',
    'wind-mill',
]

# 绘制配色（浅色系）
COLORS = [
    (173, 216, 230),
    (144, 238, 144),
    (255, 182, 193),
    (240, 230, 140),
    (255, 160, 122),
    (221, 160, 221),
    (176, 224, 230),
    (255, 228, 196),
    (152, 251, 152),
    (175, 238, 238),
]


def slice_image(image, slice_height, slice_width, overlap_ratio):
    img_width, img_height = image.size
    slices = []
    coordinates = []
    step_x = int(slice_width * (1 - overlap_ratio))
    step_y = int(slice_height * (1 - overlap_ratio))
    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (x, y, min(x + slice_width, img_width), min(y + slice_height, img_height))
            slice_img = image.crop(box)
            slices.append(slice_img)
            coordinates.append((x, y))
    return slices, coordinates


def merge_predictions(predictions, slice_coordinates, orig_image_size, slice_width, slice_height, threshold=0.30):
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    orig_height, orig_width = orig_image_size
    for i, (label, boxes, scores) in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]
        scores = np.array(scores).reshape(-1)
        valid_indices = scores > threshold
        valid_labels = np.array(label).reshape(-1)[valid_indices]
        valid_boxes = np.array(boxes).reshape(-1, 4)[valid_indices]
        valid_scores = scores[valid_indices]
        for j, box in enumerate(valid_boxes):
            box[0] = np.clip(box[0] + x_shift, 0, orig_width)
            box[1] = np.clip(box[1] + y_shift, 0, orig_height)
            box[2] = np.clip(box[2] + x_shift, 0, orig_width)
            box[3] = np.clip(box[3] + y_shift, 0, orig_height)
            valid_boxes[j] = box
        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)
    return np.array(merged_labels), np.array(merged_boxes), np.array(merged_scores)


def postprocess(labels, boxes, scores, iou_threshold=0.55):
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou

    merged_labels = []
    merged_boxes = []
    merged_scores = []
    used_indices = set()
    for i in range(len(boxes)):
        if i in used_indices:
            continue
        current_box = boxes[i]
        current_label = labels[i]
        current_score = scores[i]
        boxes_to_merge = [current_box]
        scores_to_merge = [current_score]
        used_indices.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used_indices:
                continue
            if labels[j] != current_label:
                continue
            other_box = boxes[j]
            iou = calculate_iou(current_box, other_box)
            if iou >= iou_threshold:
                boxes_to_merge.append(other_box.tolist())
                scores_to_merge.append(scores[j])
                used_indices.add(j)
        xs = np.concatenate([[box[0], box[2]] for box in boxes_to_merge])
        ys = np.concatenate([[box[1], box[3]] for box in boxes_to_merge])
        merged_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
        merged_score = max(scores_to_merge)
        merged_boxes.append(merged_box)
        merged_labels.append(current_label)
        merged_scores.append(merged_score)
    return [np.array(merged_labels)], [np.array(merged_boxes)], [np.array(merged_scores)]


def draw_boxes(image_path, labels, boxes, scores=None, thrh=0.5, path="", thickness=2):
    """
    与 GT 可视化一致的 PIL 风格绘制：浅色边框 + 顶部彩色条 + 黑字（类名 + 分数）
    """
    try:
        im = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"错误：无法读取图片 {image_path}，{e}")
        return
    drawer = ImageDraw.Draw(im)
    font = ImageFont.load_default()

    def to_py_list(x):
        try:
            return x.tolist()
        except Exception:
            return x
    boxes = to_py_list(boxes)
    labels = to_py_list(labels) if labels is not None else None
    scores = to_py_list(scores) if scores is not None else None

    idxs = list(range(len(boxes)))
    if scores is not None:
        def to_float(v):
            try:
                return float(v)
            except Exception:
                return float(v.item())
        idxs = [i for i in idxs if to_float(scores[i]) >= thrh]

    for k, i in enumerate(idxs):
        bx = boxes[i]
        x1, y1, x2, y2 = [int(round(v)) for v in bx]
        W, H = im.size
        # 边界裁剪与顺序纠正
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x2 == x1:
            x2 = min(W - 1, x1 + 1)
        if y2 == y1:
            y2 = min(H - 1, y1 + 1)

        color = COLORS[k % len(COLORS)]
        drawer.rectangle([x1, y1, x2, y2], outline=color, width=max(1, thickness))

        # 文本：类名 + 分数
        label_text = ""
        if labels is not None:
            cid = labels[i]
            try:
                idx = int(cid)
            except Exception:
                idx = int(cid.item())
            if 0 <= idx < len(AITOD_CLASSES):
                cls_name = AITOD_CLASSES[idx]
            else:
                cls_name = "Unknown"
            label_text = cls_name
        if scores is not None:
            try:
                sc = float(scores[i])
            except Exception:
                sc = float(scores[i].item())
            label_text = f"{label_text} {sc:.2f}" if label_text else f"{sc:.2f}"

        if label_text:
            if hasattr(drawer, "textbbox"):
                l, t, r, b = drawer.textbbox((0, 0), label_text, font=font)
                tw, th = (r - l), (b - t)
            elif hasattr(drawer, "textsize"):
                tw, th = drawer.textsize(label_text, font=font)
            else:
                tw, th = font.getsize(label_text)
            y_bottom = y1
            y_top = max(0, y_bottom - th - 2)
            x_left = x1
            x_right = min(W - 1, x_left + tw + 2)
            if y_bottom >= y_top and x_right > x_left:
                drawer.rectangle([x_left, y_top, x_right, y_bottom], fill=color)
            drawer.text((x1 + 1, max(0, y1 - th - 1)), label_text, fill=(0, 0, 0), font=font)

    # 保存
    base = os.path.basename(image_path)
    name, ext = os.path.splitext(base)
    out_name = f"{name}_with_boxes_aitod{ext if ext else '.jpg'}"
    if path:
        os.makedirs(path, exist_ok=True)
        output_path = os.path.join(path, out_name)
    else:
        output_path = os.path.join(os.path.dirname(image_path), out_name)
    im.save(output_path)
    print(f"[OK] 保存可视化: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AITOD 推理与可视化")
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('--img-folder', type=str, required=True, help='输入图像目录')
    parser.add_argument('--output-dir', type=str, default='./pred_vis_aitod', help='输出目录')
    parser.add_argument('--score-thresh', type=float, default=0.5, help='可视化阈值')
    parser.add_argument('--input-size', type=int, default=640, help='推理输入方形尺寸')
    parser.add_argument('-s', '--sliced', type=bool, default=False)
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)
    args = parser.parse_args()

    cfg = YAMLConfig(args.config, resume=args.resume)
    # 加载权重
    checkpoint = torch.load(args.resume, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    assert os.path.isdir(args.img_folder), f"无效图像目录: {args.img_folder}"
    os.makedirs(args.output_dir, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = [f for f in os.listdir(args.img_folder) if f.lower().endswith(exts)]

    for idx, filename in enumerate(image_files):
        image_path = os.path.join(args.img_folder, filename)
        im_pil = Image.open(image_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        transforms = T.Compose([
            T.Resize((args.input_size, args.input_size)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None].to(args.device)

        if args.sliced:
            num_boxes = args.numberofboxes
            aspect_ratio = w / h if h != 0 else 1.0
            num_cols = max(1, int(np.sqrt(num_boxes * aspect_ratio)))
            num_rows = max(1, int(num_boxes / max(1, num_cols)))
            slice_height = max(1, h // num_rows)
            slice_width = max(1, w // num_cols)
            overlap_ratio = 0.2
            slices, coordinates = slice_image(im_pil, slice_height, slice_width, overlap_ratio)

            predictions = []
            for i, slice_img in enumerate(slices):
                slice_tensor = transforms(slice_img)[None].to(args.device)
                with autocast():
                    output = model(slice_tensor, torch.tensor([[slice_img.size[0], slice_img.size[1]]]).to(args.device))
                torch.cuda.empty_cache()
                labels, boxes, scores = output
                labels = labels.cpu().detach().numpy()
                boxes = boxes.cpu().detach().numpy()
                scores = scores.cpu().detach().numpy()
                predictions.append((labels, boxes, scores))

            merged_labels, merged_boxes, merged_scores = merge_predictions(predictions, coordinates, (h, w), slice_width, slice_height)
            labels, boxes, scores = postprocess(merged_labels, merged_boxes, merged_scores)
        else:
            output = model(im_data, orig_size)
            labels, boxes, scores = output

        draw_boxes(
            image_path,
            labels=labels[0],
            boxes=boxes[0],
            scores=scores[0],
            thrh=args.score_thresh,
            path=args.output_dir
        )


if __name__ == '__main__':
    main()

