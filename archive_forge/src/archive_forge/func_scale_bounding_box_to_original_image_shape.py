from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from tqdm.auto import tqdm
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops
import wandb
def scale_bounding_box_to_original_image_shape(box: torch.Tensor, resized_image_shape: Tuple, original_image_shape: Tuple, ratio_pad: bool) -> List[int]:
    """YOLOv8 resizes images during training and the label values are normalized based on this resized shape.

    This function rescales the bounding box labels to the original
    image shape.

    Reference: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/utils/callbacks/comet.py#L105
    """
    resized_image_height, resized_image_width = resized_image_shape
    box = ops.xywhn2xyxy(box, h=resized_image_height, w=resized_image_width)
    box = ops.scale_boxes(resized_image_shape, box, original_image_shape, ratio_pad)
    box = ops.xyxy2xywh(box)
    return box.tolist()