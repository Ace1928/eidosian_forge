import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import (
def filter_masks(self, masks, iou_scores, original_size, cropped_box_image, pred_iou_thresh=0.88, stability_score_thresh=0.95, mask_threshold=0, stability_score_offset=1, return_tensors='pt'):
    """
        Filters the predicted masks by selecting only the ones that meets several criteria. The first criterion being
        that the iou scores needs to be greater than `pred_iou_thresh`. The second criterion is that the stability
        score needs to be greater than `stability_score_thresh`. The method also converts the predicted masks to
        bounding boxes and pad the predicted masks if necessary.

        Args:
            masks (`Union[torch.Tensor, tf.Tensor]`):
                Input masks.
            iou_scores (`Union[torch.Tensor, tf.Tensor]`):
                List of IoU scores.
            original_size (`Tuple[int,int]`):
                Size of the orginal image.
            cropped_box_image (`np.array`):
                The cropped image.
            pred_iou_thresh (`float`, *optional*, defaults to 0.88):
                The threshold for the iou scores.
            stability_score_thresh (`float`, *optional*, defaults to 0.95):
                The threshold for the stability score.
            mask_threshold (`float`, *optional*, defaults to 0):
                The threshold for the predicted masks.
            stability_score_offset (`float`, *optional*, defaults to 1):
                The offset for the stability score used in the `_compute_stability_score` method.
            return_tensors (`str`, *optional*, defaults to `pt`):
                If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.
        """
    if return_tensors == 'pt':
        return self._filter_masks_pt(masks=masks, iou_scores=iou_scores, original_size=original_size, cropped_box_image=cropped_box_image, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh, mask_threshold=mask_threshold, stability_score_offset=stability_score_offset)
    elif return_tensors == 'tf':
        return self._filter_masks_tf(masks=masks, iou_scores=iou_scores, original_size=original_size, cropped_box_image=cropped_box_image, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh, mask_threshold=mask_threshold, stability_score_offset=stability_score_offset)