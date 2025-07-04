import pandas
import speech_service
from keyword_extractor import *
from position_find import *
import math
from videomaker import *
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
args = sys.argv
# 检查命令行参数数量
if len(args) < 3:
    logging.error("请确保提供足够的参数：Excel 文件路径和图像文件路径")
    exit()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
s_service = speech_service.SpeechService()
dt = pandas.read_excel(args[1])
text_list = []
time_list = []
for text in dt["文本"]:
    text_list.append(text)
for t in dt["时长"]:
    time_list.append(t)
subtitles = []
for i in range(len(text_list)):
    subtexts = text_list[i].split("，")
    for subtext in subtexts:
        subtitles.append({"text": subtext, "duration": math.ceil(time_list[i] / len(subtexts))})
# 列出系统支持的语音包
available_voices = s_service.list_available_voices()
print("\n可用的语音包：")
for idx, voice in enumerate(available_voices):
    print(f"{idx + 1}. ID: {voice['id']}, 名称: {voice['name']}, 语言: {voice['languages']}")
while True:
    print("是否更改音色(Y/n)？")
    op = input()
    if op == 'Y':
        print("请输入更改的音色ID：")
        id = input()
        if s_service.set_voice_by_id(id): break
    else:break
tts_result = s_service.text_to_speech_with_fixed_time(text_list, time_list, rate=150, pitch=1)
print(f"生成的音频文件: {tts_result['audio_file']}")
image = cv2.imread(args[2])
if image is None:
    logging.error(f"无法加载图像: {image_path}")
    exit()
sentences = []
for sentence in dt["画面内容"]:
    sentences.append(extract_image_operation_info(sentence)["position"])
results = find_text_regions("image.jpg", sentences)
commandlist = []
for i in range(len(dt["画面内容"])):
    op = extract_image_operation_info(str(dt["画面内容"][i]) + str(dt["备注"][i]))
    t = int(dt["时长"][i])    
    if op["type"] == None: commandlist.append([1, [0, 0], 0, 0, t])
    else:
        commandlist.append([op["type"], [results[sentences[i]][0], results[sentences[i]][1]], op["time"], op["light"], t])
generate_video(image, commandlist, audio_path = "audios/output.wav", subtitles=subtitles)
os.remove("audios/output.wav")
