from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from tqdm.auto import tqdm
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils.ops import scale_image
import wandb
from wandb.integration.ultralytics.bbox_utils import (
def plot_sam_predictions(result: Results, prompt: Dict, table: wandb.Table) -> wandb.Table:
    result = result.to('cpu')
    image = result.orig_img[:, :, ::-1]
    image, wb_box_data = structure_prompts_and_image(image, prompt)
    image = wandb.Image(image, boxes=wb_box_data, masks={'predictions': {'mask_data': np.squeeze(result.masks.data.cpu().numpy().astype(int)), 'class_labels': {0: 'Background', 1: 'Prediction'}}})
    table.add_data(image)
    return table