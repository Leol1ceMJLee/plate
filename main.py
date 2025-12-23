import os
import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

# ---------- 常量定义 ----------
CANNY_THRESHOLD1 = 100
CANNY_THRESHOLD2 = 200
SKEW_ANGLE_THRESHOLD = -45
INTERPOLATION_METHOD = cv2.INTER_CUBIC
FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png')

# ---------- 倾斜矫正函数 ----------
def correct_skew(image):
    # 将输入的彩色图像转换为灰度图像，减少计算量并突出结构信息
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用 Canny 边缘检测算子提取图像中的边缘，便于后续轮廓检测
    edges = cv2.Canny(gray, CANNY_THRESHOLD1, CANNY_THRESHOLD2)

    # 根据边缘结果查找所有轮廓，RETR_LIST 表示提取所有轮廓，不建立层级关系
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有检测到任何轮廓，说明图像中无明显边缘，直接返回原图
    if not contours:
        return image

    # 从所有轮廓中找到面积最大的一个，通常对应车牌区域
    contour = max(contours, key=cv2.contourArea)

    # 计算该轮廓的最小外接矩形（可能是倾斜的矩形）
    rect = cv2.minAreaRect(contour)

    # 获取矩形的旋转角度
    angle = rect[-1]

    # 当角度小于 -45 度时进行修正，使得旋转角度保持在合理范围内（-45° ~ 45°）
    if angle < SKEW_ANGLE_THRESHOLD:
        angle = 90 + angle

    # 获取图像的高和宽
    (h, w) = image.shape[:2]

    # 计算图像中心点，用于旋转时指定旋转中心
    center = (w // 2, h // 2)

    # 计算旋转矩阵（M），参数分别为：旋转中心、旋转角度、缩放比例（1.0表示不缩放）
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 通过仿射变换对图像进行旋转，校正倾斜的车牌
    # flags=INTER_CUBIC 表示使用三次插值，效果更平滑；
    # borderMode=REPLICATE 表示边缘空白区域用最近像素值填充
    rotated = cv2.warpAffine(image, M, (w, h), flags=INTERPOLATION_METHOD, borderMode=cv2.BORDER_REPLICATE)

    # 返回校正后的图像
    return rotated



# 装载OCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch', det=True, rec=True)

# 设置图片路径和输出路径
'''
请在image_folder写入含有待处理图像的地址
'''
image_folder = r""
output_folder = os.path.join(image_folder, "output")
os.makedirs(output_folder, exist_ok=True)

# 遍历图片文件夹中的图片
for filename in os.listdir(image_folder):
    if not filename.lower().endswith(SUPPORTED_FORMATS):
        continue

    # 打开图片
    image_path = os.path.join(image_folder, filename)
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ 无法读取 {filename}，已跳过。")
        continue

    # ---------- 图像预处理 ----------
    # 倾斜矫正
    img = correct_skew(img)

    # ---------- OCR 检测与识别 ----------
    try:
        result = ocr.ocr(img, cls=True)
    except Exception as e:
        print(f"❌ OCR处理 {filename} 失败：{e}")
        continue

    if not result or result[0] is None or len(result[0]) == 0:
        print(f"未识别到文字：{filename}")
        continue

    boxes = [line[0] for line in result[0]]
    txts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]

    # 转换图像格式用于绘制
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_with_boxes = draw_ocr(
        Image.fromarray(img_rgb),
        boxes,
        txts,
        scores,
        font_path=FONT_PATH
    )

    # 转为BGR格式并保存
    image_with_boxes = cv2.cvtColor(np.array(image_with_boxes), cv2.COLOR_RGB2BGR)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_folder, f"result_{name}{ext}")
    cv2.imwrite(output_path, image_with_boxes)

print(f"\n✅ 所有图片已处理完成！结果保存在：{output_folder}")
