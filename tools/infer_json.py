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
import json
import time
from thop import profile, clever_format


my_thrh = 0.3

input_folder = r"/home1/wjs/datasets/AI-TOD/test/images/"
output_folder = r"/home1/wjs/rtdetr-3/rtdetr_pytorch/output/rtdetr_r50vd_hgs_aitod_1_temp_1/"

json_file = r"/home1/wjs/datasets/AI-TOD/annotations/aitod_test_v1.json"

def calculate_model_stats(model, device, input_size=(1, 3, 640, 640)):
    """
    计算模型的参数数量和FLOPs
    """
    # 创建随机输入
    dummy_input = torch.randn(input_size).to(device)
    
    # 计算FLOPs和参数数量
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    # 格式化输出
    flops, params = clever_format([flops, params], "%.3f")
    
    return flops, params

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

def save_detections_to_json(detections_list, output_json_file):
    """
    将检测结果保存为JSON文件，格式与aitodpycocotools兼容
    """
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
    
    # 为每个检测结果添加唯一ID
    for i, detection in enumerate(detections_list):
        detection["id"] = i + 1  # 添加唯一ID
    
    # 保存到新的JSON文件 - aitodpycocotools期望的是对象列表，而不是字典
    try:
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(detections_list, f, indent=4, ensure_ascii=False)
        print(f"检测结果已保存到 {output_json_file}")
        print(f"共保存了 {len(detections_list)} 个检测结果")
        
        # 打印前几个检测结果以便调试
        print("前5个检测结果示例：")
        for i, detection in enumerate(detections_list[:5]):
            print(f"  {i+1}: {detection}")
            
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")
    
    return detections_list

def cxcywh_to_xywh(cx, cy, w, h):
    """将中心点坐标格式 [cx, cy, w, h] 转换为左上角坐标格式 [x, y, w, h]"""
    x = cx - w / 2
    y = cy - h / 2
    return x, y, w, h

def main(args):
    """
    main
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
    print("模型结构:")
    print(model)
    
    # 计算模型参数数量和FLOPs
    print("\n正在计算模型参数和FLOPs...")
    flops, params = calculate_model_stats(model.model, args.device)
    print(f"模型参数数量: {params}")
    print(f"模型FLOPs: {flops}")
    
    # ===========================
    # 只对 JSON 中的图像做推理
    # ===========================
    image_folder = input_folder

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        # 创建文件名到image_id的映射
        filename_to_id = {}
        for image_info in coco_data['images']:
            filename_to_id[image_info['file_name']] = image_info['id']
        print(f"从JSON文件中加载了 {len(filename_to_id)} 个图像ID映射")
        
        # 获取类别信息
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        print(f"数据集类别信息: {categories}")
        
        # 只用 JSON 中出现的文件名，且这些文件真实存在于文件夹中
        all_files = set(f for f in os.listdir(image_folder) if f.endswith(".png"))
        image_files = [fname for fname in filename_to_id.keys() if fname in all_files]

        missing_files = [fname for fname in filename_to_id.keys() if fname not in all_files]
        if missing_files:
            print(f"警告：有 {len(missing_files)} 个 JSON 中的文件在目录中不存在，例如：{missing_files[:5]}")
        
        print(f"最终将对 {len(image_files)} 张（JSON 中指定的）图像进行推理")

    except Exception as e:
        print(f"警告：无法读取COCO JSON文件，将使用默认映射并对整个文件夹推理: {e}")
        # 回退方案：对整个文件夹推理
        image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
        filename_to_id = {}
        for i, filename in enumerate(image_files):
            filename_to_id[filename] = i + 1

    # 统计信息
    total_inference_time = 0
    total_image_count = 0
    detected_image_count = 0
    total_detections = 0
    inference_times = []

    # 存储检测结果的列表
    detections_list = []

    print(f"\n开始处理 {len(image_files)} 张图像...")

    # 预热GPU（现在设为 0，所有图像都计入统计）
    warmup_count = 0
    processed_count = 0

    for idx, filename in enumerate(image_files):
        # 构建完整的图片路径
        image_path = os.path.join(image_folder, filename)

        try:
            im_pil = Image.open(image_path).convert('RGB')
            w, h = im_pil.size
            orig_size = torch.tensor([w, h])[None].to(args.device)
            
            transforms = T.Compose([
                T.Resize((640, 640)),  
                T.ToTensor(),
            ])
            im_data = transforms(im_pil)[None].to(args.device)
            
            start_time = time.time()  # 开始计时
            
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
                    # 这里保持你原来切片时的 AMP，可以根据需要去掉
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
                # 非切片推理：不使用 autocast，避免 scatter dtype 冲突
                output = model(im_data, orig_size)
                labels, boxes, scores = output
            
            end_time = time.time()  # 结束计时
            infer_time = end_time - start_time
            
            # 统计时间（现在 warmup_count=0，所有图片都计入）
            if idx >= warmup_count:
                total_inference_time += infer_time
                inference_times.append(infer_time)
                processed_count += 1

            total_image_count += 1  # 处理图像数加一

            # 判断该图像是否有检测框（scores 中是否有值 >= 阈值）
            if scores[0] is not None and np.any(scores[0].detach().cpu().numpy() >= my_thrh):
                detected_image_count += 1
                detections_count = np.sum(scores[0].detach().cpu().numpy() >= my_thrh)
                total_detections += detections_count

            # 存储检测结果
            image_id = filename_to_id.get(filename, idx + 1)
            
            # 过滤低于阈值的检测结果并添加到列表
            if scores[0] is not None:
                scores_np = scores[0].detach().cpu().numpy()
                boxes_np = boxes[0].detach().cpu().numpy()
                labels_np = labels[0].detach().cpu().numpy()
                
                for i, score in enumerate(scores_np):
                    if score >= my_thrh:
                        box = boxes_np[i]
                
                        if box[2] > box[0] and box[3] > box[1]:  
                            # 说明是 [x1, y1, x2, y2]
                            x1, y1, x2, y2 = box
                            w = x2 - x1
                            h = y2 - y1
                            x, y = x1, y1
                        else:  
                            # 说明是 [cx, cy, w, h]
                            cx, cy, w, h = box
                            x = cx - w / 2
                            y = cy - h / 2
                
                        x = max(0, float(x))
                        y = max(0, float(y))
                        w = max(0, float(w))
                        h = max(0, float(h))
                
                        detection = {
                            "image_id": int(image_id),
                            "category_id": int(labels_np[i]) + 1,
                            "bbox": [x, y, w, h],
                            "score": float(score)
                        }
                        detections_list.append(detection)
            
            # 每处理10张图像输出一次进度
            if (idx + 1) % 10 == 0:
                print(f"已处理 {idx + 1}/{len(image_files)} 张图像")

        except Exception as e:
            print(f"处理图像 {filename} 时出错: {e}")
            continue

    # 在处理完所有图像后，保存JSON文件
    output_json = os.path.join(output_folder, "detections.json")
    save_detections_to_json(detections_list, output_json)

    # 计算性能统计
    if processed_count > 0:
        avg_time = total_inference_time / processed_count
        fps = 1.0 / avg_time
        
        # 计算标准差和置信区间
        if len(inference_times) > 1:
            time_std = np.std(inference_times)
            confidence_interval = 1.96 * time_std / np.sqrt(len(inference_times))
            min_fps = 1.0 / (avg_time + confidence_interval)
            max_fps = 1.0 / (avg_time - confidence_interval)
        else:
            time_std = 0
            min_fps = fps
            max_fps = fps

        print("\n" + "="*50)
        print("模型性能统计")
        print("="*50)
        print(f"模型参数数量: {params}")
        print(f"模型FLOPs: {flops}")
        print(f"处理图像数量(用于时间统计): {processed_count}")
        print(f"平均推理时间: {avg_time:.4f} ± {time_std:.4f} 秒")
        print(f"FPS: {fps:.2f} (范围: {min_fps:.2f} - {max_fps:.2f})")
        print(f"检测框图像数量: {detected_image_count} / {total_image_count}")
        print(f"检测框图像占比: {(detected_image_count / total_image_count) * 100:.2f}%")
        print(f"总检测框数量: {total_detections}")
        print("="*50)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=r"/home1/wjs/rtdetr-3/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_hgs_visdrone.yml")
    parser.add_argument('-r', '--resume', type=str,
                        default=r"/home1/wjs/rtdetr-3/rtdetr_pytorch/output/rtdetr_r50vd_hgs_aitod_1_temp_1/checkpoint0071.pth")
    parser.add_argument('-f', '--im-dir', type=str, )
    parser.add_argument('-s', '--sliced', type=bool, default=False)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)
    args = parser.parse_args()
    main(args)
