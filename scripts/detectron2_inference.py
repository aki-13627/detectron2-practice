import sys
import os
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import cv2
import matplotlib.pyplot as plt

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

from configs.voc_config import get_voc_config


def run_inference(image_path, model_weights):
    cfg = get_voc_config()
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    torch.load(cfg.MODEL.WEIGHTS, map_location="cpu", weights_only=True)
    predictor = DefaultPredictor(cfg)

    print(predictor.model)

    image = cv2.imread(image_path)
    outputs = predictor(image)

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
    vis_output = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))

    output_path = "output_image.jpg"
    cv2.imwrite(output_path, vis_output.get_image()[:, :, ::-1])
    print(f"推論結果を保存しました: {output_path}")

    instances = outputs["instances"]
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()

    # クラスラベルをカテゴリ名に変換
    voc_classes = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
    class_names = [voc_classes[i] for i in classes]
    print("バウンディングボックス:", boxes)
    print("スコア:", scores)
    print("カテゴリ:", class_names)

    plt.imshow(vis_output.get_image())
    plt.axis("off")
    plt.show()
    output_path = "output_image.jpg"
    vis_output.save(output_path)
    print(f"推論結果を保存しました: {output_path}")
    
    


if __name__ == "__main__":
    IMAGE_PATH = "./images/2007_000256.jpg"
    MODEL_WEIGHTS = "./output/model_final.pth"

    run_inference(IMAGE_PATH, MODEL_WEIGHTS)
