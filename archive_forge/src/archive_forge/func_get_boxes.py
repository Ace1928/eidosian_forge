from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from tqdm.auto import tqdm
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops
import wandb
def get_boxes(result: Results) -> Tuple[Dict, Dict]:
    """Convert an ultralytics prediction result into metadata for the `wandb.Image` overlay system."""
    boxes = result.boxes.xywh.long().numpy()
    classes = result.boxes.cls.long().numpy()
    confidence = result.boxes.conf.numpy()
    class_id_to_label = {int(k): str(v) for k, v in result.names.items()}
    mean_confidence_map = get_mean_confidence_map(classes, confidence, class_id_to_label)
    box_data = []
    for idx in range(len(boxes)):
        box_data.append({'position': {'middle': [int(boxes[idx][0]), int(boxes[idx][1])], 'width': int(boxes[idx][2]), 'height': int(boxes[idx][3])}, 'domain': 'pixel', 'class_id': int(classes[idx]), 'box_caption': class_id_to_label[int(classes[idx])], 'scores': {'confidence': float(confidence[idx])}})
    boxes = {'predictions': {'box_data': box_data, 'class_labels': class_id_to_label}}
    return (boxes, mean_confidence_map)