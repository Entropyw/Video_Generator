import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import re
import random
from PIL import Image
import logging
from tqdm import tqdm
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. 加载中文CLIP模型
def load_clip_model():
    model_name = "chinese-clip-vit-large-patch14-336px"  # 中文增强版
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChineseCLIPModel.from_pretrained(model_name).to(device)
    processor = ChineseCLIPProcessor.from_pretrained(model_name)
    logging.info("中文CLIP模型加载完成")
    return model, processor, device

# 2. 加载SAM模型
def load_sam_model(model_path):
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device)
    predictor = SamPredictor(sam)
    logging.info("SAM模型加载完成")
    return predictor

# 3. 分割图像并提取区域
def segment_image(image_path, predictor):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    
    # 使用SAM自动生成分割掩码
    logging.info("开始图像分割")
    masks = predictor.predict(
        point_coords=None,
        point_labels=None,
        multimask_output=True
    )[0]  # 返回所有分割掩码
    logging.info("图像分割完成")
    return image_rgb, masks

# 4. 计算区域中心坐标
def get_region_center(mask, image_shape=None):
    coords = np.where(mask)
    if len(coords[0]) == 0 and image_shape is not None:
        # 如果掩码为空，返回图像中心作为备用
        return (image_shape[1] // 2, image_shape[0] // 2)
    center_y = int(np.mean(coords[0]))
    center_x = int(np.mean(coords[1]))
    return center_x, center_y

# 5. 计算图像区域与文本的相似度
def compute_similarity(image_rgb, mask, text, model, processor, device):
    # 提取区域图像
    masked_image = image_rgb * mask[:, :, None]
    pil_image = Image.fromarray(masked_image.astype(np.uint8))
    
    # 使用CLIP处理器处理图像和文本
    inputs = processor(text=[text], images=pil_image, return_tensors="pt", padding=True).to(device)
    
    # 计算相似度
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        similarity = logits_per_image.softmax(dim=0).cpu().numpy()[0]
    return similarity

# 6. 主函数
def find_text_regions(image_path, text_descriptions, sam_model_path="sam_vit_h_4b8939.pth"):
    start_time = time.time()
    # 加载模型
    clip_model, clip_processor, device = load_clip_model()
    sam_predictor = load_sam_model(sam_model_path)
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    # 分割图像
    image_rgb, masks = segment_image(image_path, sam_predictor)
    
    # 计算每个区域与每个文本的相似度
    total_masks = len(masks)
    if total_masks == 0:
        logging.warning("没有找到任何分割区域")
        return None
    
    logging.info(f"找到 {total_masks} 个分割区域")
    
    # 初始化结果字典
    results = {text: None for text in text_descriptions}
    
    # 为每个文本描述找到最佳匹配区域
    for text_description in text_descriptions:
        max_similarity = -float('inf')
        best_center = None
        if text_description == "":
            logging.warning("未找到文本，返回随机区域")
            best_center = (random.randint(0, width), random.randint(0, height))
        elif re.search("中|左|右|上|下", text_description):
            logging.info("找到方位关键词")
            center_now = [width // 2, height // 2]
            if re.search("左", text_description): center_now[0] = max(0, center_now[0] - width // 8)
            if re.search("右", text_description): center_now[0] = min(width, center_now[0] + width // 8)
            if re.search("上", text_description): center_now[1] = max(0, center_now[1] - height // 8)
            if re.search("下", text_description): center_now[1] = min(width, center_now[1] + height // 8)
            best_center = center_now.copy()
        else:
            # 使用 tqdm 创建进度条
            for i in tqdm(range(total_masks), desc=f"处理 {text_description[:20]}...", unit="区域"):
                mask = masks[i]
                similarity = compute_similarity(image_rgb, mask, text_description, clip_model, clip_processor, device)
                if similarity > max_similarity:
                    max_similarity = similarity
                    center = get_region_center(mask, image_rgb.shape)
                    best_center = center
                
                # 计算当前进度百分比并记录日志（每10%记录一次）
                progress_percent = (i + 1) / total_masks * 100
                if progress_percent % 10 == 0:
                    logging.info(f"处理进度: {progress_percent:.1f}%")
        
        # 如果没有有效掩码，返回图像中心
        if best_center is None:
            best_center = (image_rgb.shape[1] // 2, image_rgb.shape[0] // 2)
            logging.warning(f"文本 '{text_description}' 未找到匹配区域，返回图像中心点")
        
        results[text_description] = best_center
        logging.info(f"文本 '{text_description}' 对应的区域中心坐标: ({best_center[0]}, {best_center[1]})")
    end_time = time.time()
    
    logging.info(f"总耗时: {end_time - start_time:.2f} 秒")
    for text, center in results.items():
        logging.info(f"文本 '{text}' 对应的区域中心坐标: ({center[0]}, {center[1]})")
    return results
'''
if __name__ == "__main__":
    image_path = "image.jpg"  # 替换为你的图片路径
    text_descriptions = [
        "海面平静的部分",
        "海浪轻柔拍打礁石的画面",
        "整体壮阔",
        "海天相接处，海面与礁石的交界，展现连接与广阔。",
    ]  # 替换为你的文字描述列表
'''
