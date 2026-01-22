from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, is_torch_available, logging, requires_backends

        Converts the output of [`SegGptImageSegmentationOutput`] into segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`SegGptImageSegmentationOutput`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
            num_labels (`int`, *optional*):
                Number of classes in the segmentation task (excluding the background). If specified, a palette will be
                built, assuming that class_idx 0 is the background, to map prediction masks from RGB values to class
                indices. This value should be the same used when preprocessing inputs.
        Returns:
            semantic_segmentation: `List[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.
        