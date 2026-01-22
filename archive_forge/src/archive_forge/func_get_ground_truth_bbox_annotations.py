from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from tqdm.auto import tqdm
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops
import wandb
def get_ground_truth_bbox_annotations(img_idx: int, image_path: str, batch: Dict, class_name_map: Dict=None) -> List[Dict[str, Any]]:
    """Get ground truth bounding box annotation data in the form required for `wandb.Image` overlay system."""
    indices = batch['batch_idx'] == img_idx
    bboxes = batch['bboxes'][indices]
    cls_labels = batch['cls'][indices].squeeze(1).tolist()
    class_name_map_reverse = {v: k for k, v in class_name_map.items()}
    if len(bboxes) == 0:
        wandb.termwarn(f'Image: {image_path} has no bounding boxes labels', repeat=False)
        return None
    cls_labels = batch['cls'][indices].squeeze(1).tolist()
    if class_name_map:
        cls_labels = [str(class_name_map[label]) for label in cls_labels]
    original_image_shape = batch['ori_shape'][img_idx]
    resized_image_shape = batch['resized_shape'][img_idx]
    ratio_pad = batch['ratio_pad'][img_idx]
    data = []
    for box, label in zip(bboxes, cls_labels):
        box = scale_bounding_box_to_original_image_shape(box, resized_image_shape, original_image_shape, ratio_pad)
        data.append({'position': {'middle': [int(box[0]), int(box[1])], 'width': int(box[2]), 'height': int(box[3])}, 'domain': 'pixel', 'class_id': class_name_map_reverse[label], 'box_caption': label})
    return data