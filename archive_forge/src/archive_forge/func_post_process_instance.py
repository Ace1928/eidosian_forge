import io
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def post_process_instance(self, results, outputs, orig_target_sizes, max_target_sizes, threshold=0.5):
    """
        Converts the output of [`DetrForSegmentation`] into actual instance segmentation predictions. Only supports
        PyTorch.

        Args:
            results (`List[Dict]`):
                Results list obtained by [`~DetrImageProcessor.post_process`], to which "masks" results will be added.
            outputs ([`DetrSegmentationOutput`]):
                Raw outputs of the model.
            orig_target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (h, w) of each image of the batch. For evaluation, this must be the original
                image size (before any data augmentation).
            max_target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the maximum size (h, w) of each image of the batch. For evaluation, this must be the
                original image size (before any data augmentation).
            threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels, boxes and masks for an
            image in the batch as predicted by the model.
        """
    logger.warning_once('`post_process_instance` is deprecated and will be removed in v5 of Transformers, please use `post_process_instance_segmentation`.')
    if len(orig_target_sizes) != len(max_target_sizes):
        raise ValueError('Make sure to pass in as many orig_target_sizes as max_target_sizes')
    max_h, max_w = max_target_sizes.max(0)[0].tolist()
    outputs_masks = outputs.pred_masks.squeeze(2)
    outputs_masks = nn.functional.interpolate(outputs_masks, size=(max_h, max_w), mode='bilinear', align_corners=False)
    outputs_masks = (outputs_masks.sigmoid() > threshold).cpu()
    for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
        img_h, img_w = (t[0], t[1])
        results[i]['masks'] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
        results[i]['masks'] = nn.functional.interpolate(results[i]['masks'].float(), size=tuple(tt.tolist()), mode='nearest').byte()
    return results