import re

def extract_image_operation_info(input_sentence):
    result = {
        "type": None,
        "position": "",
        "time": 2,
        "light": 0
    }
    operation_type_patterns = {
        0: "聚焦|集中|对焦|特写|缩放|拉近",
        1: "拉远|远离|退远|拉出|推出|扩展|放大|拉宽|拉广|整体视图|全景视图|远景视图|宽景视图|广角视图",
        2: "移动|滑动|平移|滑行|变动|挪动|位移|迁移|转移|行进|前进|后退|左移|右移|上移|下移",
    }
    speed_patterns = {
        4: "缓缓|缓慢|慢慢|慢速|迟缓|徐缓|怠缓|舒缓|迟钝|滞缓|缓行|缓进|缓移|缓动|缓行|缓进|缓移|缓动|逐渐",
        1: "快速|迅速"
    }
    position_patterns = "于|到|至|直到"
    for op_type, pattern in operation_type_patterns.items():
        if re.search(pattern, input_sentence):
            result["type"] = op_type
            break
    if re.search("亮度", input_sentence):
        if re.search("提高|调亮|调高|变高|增加|变亮", input_sentence):
            result["light"] = 1
        elif re.search("减少|降低|调暗|变暗|调低|变低", input_sentence):
            result["light"] = 2
    # 操作位置提取（保留原文描述）
    match = re.search(position_patterns, input_sentence)
    if match:
        result["position"] = input_sentence[match.end():]

    if re.search("中间", input_sentence):
        result["position"] = "中间"
        if re.search("左", input_sentence): result["position"] += "左"
        if re.search("右", input_sentence): result["position"] += "右"
        if re.search("上", input_sentence): result["position"] += "上"
        if re.search("下", input_sentence): result["position"] += "下"
    elif re.search("左|右|上|下", input_sentence): 
        result["position"] = ""
        if re.search("左", input_sentence): result["position"] += "左"
        if re.search("右", input_sentence): result["position"] += "右"
        if re.search("上", input_sentence): result["position"] += "上"
        if re.search("下", input_sentence): result["position"] += "下"
    for speed_level, pattern in speed_patterns.items():
        if re.search(pattern, input_sentence):
            result["time"] = speed_level
            break
    
    return result
'''
# 测试示例
input_sentence_1 = "缓慢推入《海天情长》这幅画，并聚焦于海面平静的部分"
input_sentence_2 = "镜头缓缓移动，展现《海天情长》这幅画中海浪轻柔拍打礁石的画面，强调“悄悄积蓄”的力量感，而非猛烈冲击。"
input_sentence_3 = "画面的亮度可以适当提高，结尾处给人以希望和开阔感"
print(extract_image_operation_info(input_sentence_2))
'''
