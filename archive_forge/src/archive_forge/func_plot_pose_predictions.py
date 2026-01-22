from typing import Any, Optional
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from ultralytics.engine.results import Results
from ultralytics.models.yolo.pose import PosePredictor
from ultralytics.utils.plotting import Annotator
import wandb
from wandb.integration.ultralytics.bbox_utils import (
def plot_pose_predictions(result: Results, model_name: str, visualize_skeleton: bool, table: Optional[wandb.Table]=None):
    result = result.to('cpu')
    boxes, mean_confidence_map = get_boxes(result)
    annotated_image = annotate_keypoint_results(result, visualize_skeleton)
    prediction_image = wandb.Image(annotated_image, boxes=boxes)
    table_row = [model_name, prediction_image, len(boxes['predictions']['box_data']), mean_confidence_map, result.speed]
    if table is not None:
        table.add_data(*table_row)
        return table
    return table_row