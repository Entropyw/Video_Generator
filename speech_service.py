import os
import time
import logging
import uuid
import pyttsx3
import numpy as np
from scipy.io.wavfile import write, read
import threading

class SpeechService:
    # 语音服务核心类，提供文字转语音功能
    def __init__(self):
        # 初始化日志
        self._setup_logging()
        
        # 初始化 TTS 引擎
        self.tts_engine = pyttsx3.init()
        
        # 获取系统支持的语音包
        self.voices = self.tts_engine.getProperty('voices')
        self.tts_lock = threading.Lock()
        # 设置默认中文音色（如果可用）
        self._set_chinese_voice()
        
        # 创建音频文件存储目录
        self.audio_files_dir = "audios"
        os.makedirs(self.audio_files_dir, exist_ok=True)

    def _setup_logging(self):
        # 配置日志系统，使用中文格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("speech_service")

    def _set_chinese_voice(self):
        # 尝试设置中文音色
        for voice in self.voices:
            if 'chinese' in voice.name.lower() or '中文' in voice.name or 'cmn' in voice.languages:
                self.tts_engine.setProperty('voice', voice.id)
                self.logger.info(f"已设置中文音色：{voice.name}")
                return
        self.logger.warning("未找到中文音色，使用默认音色。")

    def text_to_speech_with_fixed_time(self, text_list: list, times_list: list, rate: int = 200, pitch: float = 1.0) -> dict:
        # 根据文本列表和时间列表生成固定时长的音频文件
        try:
            if len(text_list) != len(times_list):
                raise ValueError("文本列表与时间列表长度必须相同")

            audio_clips = []
            intermediate_files = []

            # 生成单个音频片段
            for text in text_list:
                audio_path = self.text_to_speech(text, rate)
                audio_clips.append(audio_path)
                intermediate_files.append(audio_path)

            # 合并音频文件并调整时长
            combined_audio_path = os.path.join(self.audio_files_dir, f"combined_{uuid.uuid4().hex}.wav")
            self._combine_audio_files(audio_clips, times_list, combined_audio_path)
            intermediate_files.append(combined_audio_path)

            # 调整音调并保存到固定输出文件
            output_path = os.path.join(self.audio_files_dir, "output.wav")
            self._adjust_pitch(combined_audio_path, output_path, pitch)

            # 清理中间文件
            for file_path in intermediate_files:
                if os.path.exists(file_path):
                    os.remove(file_path)

            return {
                "success": True,
                "message": "音频文件生成成功",
                "audio_file": output_path
            }
        except Exception as e:
            self.logger.error(f"文本转语音错误：{str(e)}")
            return {
                "success": False,
                "message": f"文本转语音错误：{str(e)}",
                "audio_file": None
            }

    def text_to_speech(self, text: str, rate: int = 200) -> str:
        # 将文本转换为语音并保存为 WAV 文件
        try:
            text = text.strip()
            if not text:
                raise ValueError("文本不能为空")
            with self.tts_lock:
                self.tts_engine.setProperty('rate', rate)
                audio_filename = f"tts_{int(time.time())}_{uuid.uuid4().hex}.wav"
                audio_filepath = os.path.join(self.audio_files_dir, audio_filename)
            
                self.tts_engine.save_to_file(text, audio_filepath)
                self.tts_engine.runAndWait()
            
                if not os.path.exists(audio_filepath):
                    raise RuntimeError("无法创建音频文件")
            
                self.logger.info(f"文本已转换为语音并保存至：{audio_filepath}")
                return audio_filepath
        except Exception as e:
            self.logger.error(f"文本转语音错误：{str(e)}")
            raise RuntimeError(f"文本转语音错误：{str(e)}")

    def _combine_audio_files(self, audio_clips: list, times_list: list, output_path: str):
        # 合并多个音频文件并根据需要添加静默
        try:
            combined_audio = []
            sample_rate = None

            for clip, duration in zip(audio_clips, times_list):
                # 读取音频文件
                sr, audio_data = read(clip)
                
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    raise ValueError(f"采样率不匹配：{sr} != {sample_rate}")

                # 确保音频为单声道，若为立体声则转换为单声道
                if audio_data.ndim == 2:
                    audio_data = audio_data.mean(axis=1).astype(np.int16)

                combined_audio.append(audio_data)

                # 计算音频时长并添加静默
                audio_duration = len(audio_data) / sample_rate

                if audio_duration < duration:
                    silence_samples = int((duration - audio_duration) * sample_rate)
                    silence_array = np.zeros(silence_samples, dtype=np.int16)
                    combined_audio.append(silence_array)

            # 拼接所有音频数据
            combined_audio = np.concatenate(combined_audio)
            write(output_path, sample_rate, combined_audio)
            self.logger.info(f"合并后的音频已保存至：{output_path}")

        except Exception as e:
            self.logger.error(f"合并音频文件错误：{str(e)}")
            raise RuntimeError(f"合并音频文件错误：{str(e)}")

    def _adjust_pitch(self, input_path: str, output_path: str, pitch: float):
        # 调整音频音调
        try:
            # 读取输入音频
            sample_rate, audio_data = read(input_path)
            
            # 确保音频为单声道
            if audio_data.ndim == 2:
                audio_data = audio_data.mean(axis=1).astype(np.int16)

            # 使用重采样调整音调
            from scipy.interpolate import interp1d
            original_length = len(audio_data)
            new_length = int(original_length / pitch)
            x = np.linspace(0, 1, original_length)
            x_new = np.linspace(0, 1, new_length)
            interpolator = interp1d(x, audio_data, kind='linear')
            adjusted_audio = interpolator(x_new).astype(np.int16)

            # 保存调整后的音频
            write(output_path, sample_rate, adjusted_audio)
            self.logger.info(f"音调调整后的音频已保存至：{output_path}")
        except Exception as e:
            self.logger.error(f"音调调整错误：{str(e)}")
            raise RuntimeError(f"音调调整错误：{str(e)}")

    def list_available_voices(self):
        # 列出系统支持的语音包
        available_voices = []
        for voice in self.voices:
            available_voices.append({
                "id": voice.id,
                "name": voice.name,
                "languages": voice.languages
            })
        return available_voices

    def set_voice_by_id(self, voice_id: str):
        # 根据语音包 ID 设置音色
        for voice in self.voices:
            if voice.id == voice_id:
                self.tts_engine.setProperty('voice', voice.id)
                self.logger.info(f"已设置音色为：{voice.name}")
                return True
        self.logger.warning(f"未找到 ID 为 {voice_id} 的音色。")
        return False

if __name__ == "__main__":
    # 初始化服务
    speech_service = SpeechService()
    
    # 列出系统支持的语音包
    available_voices = speech_service.list_available_voices()
    print("\n可用的语音包：")
    for idx, voice in enumerate(available_voices):
        print(f"{idx + 1}. ID: {voice['id']}, 名称: {voice['name']}, 语言: {voice['languages']}")
    
    # 测试文本转语音
    print("\n文本转语音测试：")
    text_list = ["你好，这是一个测试文本", "这是第二个测试文本"]
    times_list = [3.0, 5.0]
    tts_result = speech_service.text_to_speech_with_fixed_time(text_list, times_list, rate=150, pitch=1.2)
    if tts_result["success"]:
        print(f"生成的音频文件：{tts_result['audio_file']}")
    else:
        print(f"错误：{tts_result['message']}")