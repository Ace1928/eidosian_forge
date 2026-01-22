from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from tqdm.auto import tqdm
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils.ops import scale_image
import wandb
from wandb.integration.ultralytics.bbox_utils import (
def structure_prompts_and_image(image: np.array, prompt: Dict) -> Dict:
    wb_box_data = []
    if prompt['bboxes'] is not None:
        wb_box_data.append({'position': {'middle': [prompt['bboxes'][0], prompt['bboxes'][1]], 'width': prompt['bboxes'][2], 'height': prompt['bboxes'][3]}, 'domain': 'pixel', 'class_id': 1, 'box_caption': 'Prompt-Box'})
    if prompt['points'] is not None:
        image = image.copy().astype(np.uint8)
        image = cv2.circle(image, tuple(prompt['points']), 5, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    wb_box_data = {'prompts': {'box_data': wb_box_data, 'class_labels': {1: 'Prompt-Box'}}}
    return (image, wb_box_data)