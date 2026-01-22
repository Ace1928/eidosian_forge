from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from tqdm.auto import tqdm
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops
import wandb
def plot_bbox_predictions(result: Results, model_name: str, table: Optional[wandb.Table]=None) -> Union[wandb.Table, Tuple[wandb.Image, Dict, Dict]]:
    """Plot the images with the W&B overlay system.

    The `wandb.Image` is either added to a `wandb.Table` or returned.
    """
    result = result.to('cpu')
    boxes, mean_confidence_map = get_boxes(result)
    image = wandb.Image(result.orig_img[:, :, ::-1], boxes=boxes)
    if table is not None:
        table.add_data(model_name, image, len(boxes['predictions']['box_data']), mean_confidence_map, result.speed)
        return table
    return (image, boxes['predictions'], mean_confidence_map)