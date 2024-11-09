import argparse
import cv2
import os
import xml.etree.ElementTree as ET
from ultralytics import YOLO


def load_model(model_path):
    model = YOLO(model_path)  # 使用 YOLO 类加载模型
    return model


def detect(model, image_path):
    img = cv2.imread(image_path)
    results = model(img)  # 进行目标检测
    return img, results


def save_as_xml(results, output_path):
    root = ET.Element("detections")

    # results 是一个列表，包含检测结果
    for result in results:
        boxes = result.boxes  # 获取检测到的框
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # 获取边界框坐标
            conf = box.conf[0]  # 获取置信度
            cls = box.cls[0]  # 获取类别

            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "class").text = str(int(cls))
            ET.SubElement(obj, "confidence").text = str(conf.item())
            ET.SubElement(obj, "xmin").text = str(int(x1))
            ET.SubElement(obj, "ymin").text = str(int(y1))
            ET.SubElement(obj, "xmax").text = str(int(x2))
            ET.SubElement(obj, "ymax").text = str(int(y2))

    tree = ET.ElementTree(root)
    tree.write(output_path)


def main(args):
    model = load_model(args.model)
    img, results = detect(model, args.dir)
    save_as_xml(results, args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', type=str, required=True, help='Path to the input image file')
    parser.add_argument('-model', type=str, required=True, help='Path to the YOLOv8 model')
    parser.add_argument('-out', type=str, required=True, help='Path to the output XML file')

    args = parser.parse_args()

    main(args)
