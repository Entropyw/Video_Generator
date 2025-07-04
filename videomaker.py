import cv2
import numpy as np
import os
import imageio.v2 as imageio
import moviepy as mp
import logging
import time
from tqdm import tqdm

def adjust_image_size(image):
    height, width = image.shape[:2]
    new_width = ((width + 15) // 32) * 32
    new_height = ((height + 15) // 32) * 32
    return cv2.resize(image, (new_width, new_height))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def focus(image, pos, scale):
    height, width = image.shape[:2]
    center_x = int(pos[0] * width)
    center_y = int(pos[1] * height)
    transform_matrix = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
    transformed_image = cv2.warpAffine(image, transform_matrix, (width, height))
    return cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)

image_path = "image.jpg"
output_video = "output.mp4"

def generate_video(image, commandlist, audio_path, subtitles):
    start_time = time.time()
    scale = 0.5
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    image = adjust_image_size(image)
    height, width = image.shape[:2]
    pos = [width // 2, height // 2]
    fps = 24
    total_frames = 0
    duration = 0
   
    for command in commandlist:
        total_frames += int(max(command[2], command[4]) * 24)
        duration += max(command[2], command[4])
    if audio_path:
        audio_duration = mp.AudioFileClip(audio_path).duration
        if audio_duration > duration: total_frames += int((audio_duration - duration) * 24)
    writer = imageio.get_writer("temp.mp4", fps=fps, codec='libx264')
    image_now = image.copy()
    progress_bar = tqdm(total=total_frames, desc="视频帧生成进度", unit="帧")
    
    for command in commandlist:
        if command[3] == 1:
            image_now = cv2.convertScaleAbs(image, alpha=1.10, beta=0)
        elif command[3] == 2:
            image_now = cv2.convertScaleAbs(image, alpha=0.90, beta=0)
        if command[0] == 0:
            op = (1 / scale) ** (1 / (command[2] * 24))
            cnt = 0
            pos = command[1].copy()
            pos[0] = max(width // 4, min(pos[0], width * 3 // 4))
            pos[1] = max(min(width // 6, height // 2), min(pos[1], height - min(width // 6, height // 2)))
            left = max(0, pos[0] - width // 4)
            right = min(width, pos[0] + width // 4)
            top = max(0, pos[1] - min(width // 6, height // 2))
            bottom = min(height, pos[1] + min(width // 6, height // 2))
            cropped_image = image_now[int(top):int(bottom), int(left):int(right)]
            while cnt < int(command[2] * 24):
                scale *= op
                cnt += 1
                transformed_image = focus(image_now, (pos[0] / width, pos[1] / height), scale)
                cropped_image = transformed_image[int(top):int(bottom), int(left):int(right)]
                writer.append_data(cropped_image)
                progress_bar.update(1)
            while cnt < int(command[4] * 24):
                cnt += 1
                writer.append_data(cropped_image)
                progress_bar.update(1)
        
        elif command[0] == 1:
            op = (0.25 / scale) ** (1 / (command[2] * 24))
            cnt = 0
            pos = command[1].copy()
            pos[0] = max(width // 4, min(pos[0], width * 3 // 4))
            pos[1] = max(min(width // 6, height // 2), min(pos[1], height - min(width // 6, height // 2)))
            left = max(0, pos[0] - width // 4)
            right = min(width, pos[0] + width // 4)
            top = max(0, pos[1] - min(width // 6, height // 2))
            bottom = min(height, pos[1] + min(width // 6, height // 2))
            cropped_image = image_now[int(top):int(bottom), int(left):int(right)]
            while cnt < int(command[2] * 24):
                scale *= op
                cnt += 1
                transformed_image = focus(image_now, (pos[0] / width, pos[1] / height), scale)
                cropped_image = transformed_image[int(top):int(bottom), int(left):int(right)]
                writer.append_data(cropped_image)
                progress_bar.update(1)
            while cnt < int(command[4] * 24):
                cnt += 1
                writer.append_data(cropped_image)
                progress_bar.update(1)
        
        elif command[0] == 2:
            target_pos = command[1]
            if abs(target_pos[0] - pos[0]) < width // 4 and abs(target_pos[1] - pos[1]) < width // 4:
                if pos[0] < width // 2: target_pos[0] = pos[0] + width // 4
                else: target_pos[0] = pos[0] - width // 4
                if pos[1] < width // 2: target_pos[1] = pos[1] + width // 4
                else: target_pos[1] = pos[1] - width // 4
            add_x = (target_pos[0] - pos[0]) / (command[2] * 1000 / 24)
            add_y = (target_pos[1] - pos[1]) / (command[2] * 1000 / 24)
            cnt = 0
            cropped_image = image_now[int(top):int(bottom), int(left):int(right)]
            while cnt < int(command[2] * 24):
                cnt += 1
                pos[0] += add_x
                pos[1] += add_y
                pos[0] = int(pos[0])
                pos[1] = int(pos[1])
                pos[0] = max(width // 4, min(pos[0], width * 3 // 4))
                pos[1] = max(min(width // 6, height // 2), min(pos[1], height - min(width // 6, height // 2)))
                transformed_image = focus(image_now, (pos[0] / width, pos[1] / height), scale)
                left = max(0, pos[0] - width // 4)
                right = min(width, pos[0] + width // 4)
                top = max(0, pos[1] - min(width // 6, height // 2))
                bottom = min(height, pos[1] + min(width // 6, height // 2))
                cropped_image = transformed_image[int(top):int(bottom), int(left):int(right)]
                writer.append_data(cropped_image)
                progress_bar.update(1)
            while cnt < int(command[4] * 24):
                cnt += 1
                writer.append_data(cropped_image)
                progress_bar.update(1)
    while progress_bar.n < total_frames:
        writer.append_data(cropped_image)
        progress_bar.update(1)
    writer.close()
    progress_bar.close()
    if audio_path:
        try:
            logging.info("正在向视频中添加音频")
            video_clip = mp.VideoFileClip("temp.mp4")
            audio_clip = mp.AudioFileClip(audio_path)
            final_clip = video_clip.with_audio(audio_clip)
            temp_final = "temp_with_audio.mp4"
            final_clip.write_videofile(temp_final, codec='libx264', audio_codec='aac')
            # 添加字幕
            logging.info("开始添加字幕")
            
            # 创建视频剪辑列表
            video_clips = [mp.VideoFileClip(temp_final)]
            current_time = 0
            # 为每个字幕创建文本剪辑并与视频剪辑组合
            for subtitle in subtitles:
                text_clip = mp.TextClip(
                    font="font.ttf",
                    text=subtitle["text"],
                    font_size=40,
                    size=(video_clip.size[0], 100), 
                    color='white', 
                    bg_color='black', 
                    method='caption'
                )
                text_clip = text_clip.with_duration(subtitle["duration"]).with_position(("center", "bottom")).with_start(current_time)
                current_time += subtitle["duration"]
                video_clips.append(text_clip)
            
            # 合并视频剪辑和字幕剪辑
            result = mp.CompositeVideoClip(video_clips)
            result.write_videofile(output_video, codec='libx264', audio_codec='aac')
            
            # 删除临时文件
            os.remove("temp.mp4")
            os.remove(temp_final)
            logging.info("字幕已成功添加到视频")
        except Exception as e:
            logging.error(f"添加音频或字幕时发生错误: {str(e)}")
    else:
        # 无音频处理
        logging.info("未提供音频文件，直接生成无音频视频")
        os.rename("temp.mp4", output_video)

    end_time = time.time()
    logging.info(f"视频已成功保存到: {output_video}")
    logging.info(f"总耗时: {end_time - start_time:.2f} 秒")
'''
if __name__ == "__main__":
    image = cv2.imread("image.jpg")
    commandlist = [
        [0, [1467, 2000], 1, 0, 2],  
        [2, [2910, 2098], 1, 0, 4],  
        [1, [0, 0], 1, 0, 4],        
        [0, [3000, 1900], 1, 1, 4]   
    ]
    subs = [
        {"text": "第一个字幕", "duration": 2},
        {"text": "第二个字幕", "duration": 3},
        {"text": "第三个字幕", "duration": 3},
        {"text": "结束", "duration": 2}
    ]
    generate_video(image, commandlist, audio_path="audios/output.wav", subtitles=subs)
    '''