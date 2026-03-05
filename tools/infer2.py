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
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
import numpy as np
import cv2

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

# 定义浅色系的颜色列表
COLORS = [
    (173, 216, 230),  # 浅蓝色
    (144, 238, 144),  # 浅绿色
    (255, 182, 193),  # 浅粉色
    (240, 230, 140),  # 浅黄色
    (255, 160, 122),  # 浅橙色
    (221, 160, 221),  # 浅紫色
    (176, 224, 230),  # 浅青色
    (255, 228, 196),  # 浅棕色
    (152, 251, 152),  # 浅薄荷绿
    (175, 238, 238),  # 浅天蓝色
]

COCO_CLASSES = [
    'pedestrian', 
    'people', 
    'bicycle', 
    'car', 
    'van', 
    'truck', 
    'tricycle',
    'awning-tricycle', 
    'bus', 
    'motor'
]

# def draw(images, labels, boxes, scores, thrh = 0.6, path = ""):
#     for i, im in enumerate(images):
#         draw = ImageDraw.Draw(im)
#         scr = scores[i]
#         lab = labels[i][scr > thrh]
#         box = boxes[i][scr > thrh]
#         scrs = scores[i][scr > thrh]
#         for j,b in enumerate(box):
#             draw.rectangle(list(b), outline=COLORS[j%10],)
#             # draw.text((b[0], b[1]), text=f"label: {lab[j].item()} {round(scrs[j].item(),2)}", font=ImageFont.load_default(), fill=COLORS[j%10])
#             draw.text((b[0], b[1]), text=f"{COCO_CLASSES[lab[j].item()]} : {round(scrs[j].item(),2)}", font=ImageFont.load_default(), fill=COLORS[j%10])
#         if path == "":
#             im.save(f'results_{i}.jpg')
#         else:
#             im.save(path)
            

def draw_boxes(image_path, labels, boxes, scores=None, thrh=0.5, path = "", thickness=2):
    """
    在图片上绘制边界框。

    参数:
    - image_path: 图片路径。
    - boxes: 边界框列表，格式为[[x1, y1, x2, y2], ...]，其中(x1, y1)是左上角，(x2, y2)是右下角。
    - labels: 可选的类别标签列表（COCO ID）。
    - scores: 可选的置信度分数列表。
    - thickness: 边界框线条粗细，默认为2。
    """
    # 读取图片
    image = cv2.imread(image_path)

    if image is None:
        print(f"错误：无法读取图片 {image_path}")
        return

    # 过滤掉 scores < 0.5 的边界框、标签和分数
    if scores is not None:
        filtered_indices = [i for i, score in enumerate(scores) if score.item() >= thrh]
        boxes = [boxes[i] for i in filtered_indices]
        if labels is not None:
            labels = [labels[i] for i in filtered_indices]
        scores = [scores[i] for i in filtered_indices]

    # 遍历所有边界框
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        # 根据索引选择颜色
        color = COLORS[i % len(COLORS)]

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # 如果有标签和分数，则在框上方显示类别名称和置信度
        if labels is not None:
            class_id = labels[i]  # COCO 类别 ID
            # print(class_id.item())
            # print(COCO_CLASSES[class_id.item()])
            class_name = COCO_CLASSES[class_id.item()] if 0 <= class_id < len(COCO_CLASSES) else "Unknown"

            label_text = class_name
            if scores is not None:
                label_text += f" {scores[i]:.2f}"

            # 调整字体大小和位置
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4  # 字体大小
            font_thickness = 1  # 字体厚度
            text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
            text_w, text_h = text_size

            # 绘制背景色
            cv2.rectangle(image, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)  # 背景色与边界框颜色一致

            # 绘制文本
            cv2.putText(image, label_text, (x1, y1 - 5), font, font_scale, (0, 0, 0),
                        font_thickness)  # 字体颜色为黑色

    # 显示图片
    # cv2.imshow("Image with Boxes", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 保存结果（可选）
    output_path = image_path.replace(".jpg", "_with_boxes_rtdetr.jpg")
    cv2.imwrite(output_path, image)
    print(f"结果已保存到 {output_path}")

def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(args.device)
    
    
    # 定义文件夹路径
    image_folder = r'/home1/lxt/PyCharmProjects/Detection/rtdetr_pytorch_Original/VisDrone_visual_figure_original'

    # 获取文件夹中所有 .jpg 文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

    # 假设 results 是一个列表，每个元素对应一张图片的检测结果
    # results 的长度应与 image_files 的长度一致
    for idx, filename in enumerate(image_files):
        # 构建完整的图片路径
        image_path = os.path.join(image_folder, filename)

        # print(image_path[-5])
        if '0' <= image_path[-5] <= '9': 
            
            # # 获取当前图片的检测结果
            # if idx < len(results):  # 确保 results 中有对应的检测结果
            #     boxes = results[idx]['boxes']
            #     labels = results[idx]['labels']
            #     scores = results[idx]['scores']

            #     # 绘制边界框
            #     draw_boxes(image_path, boxes, labels, scores)
            # else:
            #     print(f"警告：未找到文件 {filename} 的检测结果")
        
        
            
            im_pil = Image.open(image_path).convert('RGB')
            w, h = im_pil.size
            orig_size = torch.tensor([w, h])[None].to(args.device)
            
            transforms = T.Compose([
                T.Resize((640, 640)),  
                T.ToTensor(),
            ])
            im_data = transforms(im_pil)[None].to(args.device)
            
            if args.sliced:
                num_boxes = args.numberofboxes
                
                aspect_ratio = w / h
                num_cols = int(np.sqrt(num_boxes * aspect_ratio)) 
                num_rows = int(num_boxes / num_cols)
                slice_height = h // num_rows
                slice_width = w // num_cols
                overlap_ratio = 0.2
                slices, coordinates = slice_image(im_pil, slice_height, slice_width, overlap_ratio)
                predictions = []
                for i, slice_img in enumerate(slices):
                    slice_tensor = transforms(slice_img)[None].to(args.device)
                    with autocast():  # Use AMP for each slice
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
            
            # print("labels: ", labels)
            # print("scores: ", scores)
            
            draw_boxes(image_path, labels=labels[0], boxes=boxes[0], scores=scores[0], thrh=0.5)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    # parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('-f', '--im-dir', type=str, )
    parser.add_argument('-s', '--sliced', type=bool, default=False)
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)
    args = parser.parse_args()
    main(args)











