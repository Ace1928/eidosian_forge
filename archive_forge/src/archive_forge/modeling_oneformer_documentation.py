import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig

        text_inputs (`List[torch.Tensor]`, *optional*):
            Tensor fof shape `(num_queries, sequence_length)` to be fed to a model
        mask_labels (`List[torch.Tensor]`, *optional*):
            List of mask labels of shape `(num_labels, height, width)` to be fed to a model
        class_labels (`List[torch.LongTensor]`, *optional*):
            list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the
            labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.

        Returns:
            `OneFormerUniversalSegmentationOutput`
        Example:

        Universal segmentation example:

        ```python
        >>> from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> # load OneFormer fine-tuned on ADE20k for universal segmentation
        >>> processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        >>> model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")

        >>> url = (
        ...     "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
        ... )
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # Semantic Segmentation
        >>> inputs = processor(image, ["semantic"], return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # you can pass them to processor for semantic postprocessing
        >>> predicted_semantic_map = processor.post_process_semantic_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]
        >>> f"👉 Semantic Predictions Shape: {list(predicted_semantic_map.shape)}"
        '👉 Semantic Predictions Shape: [512, 683]'

        >>> # Instance Segmentation
        >>> inputs = processor(image, ["instance"], return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # you can pass them to processor for instance postprocessing
        >>> predicted_instance_map = processor.post_process_instance_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]["segmentation"]
        >>> f"👉 Instance Predictions Shape: {list(predicted_instance_map.shape)}"
        '👉 Instance Predictions Shape: [512, 683]'

        >>> # Panoptic Segmentation
        >>> inputs = processor(image, ["panoptic"], return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # you can pass them to processor for panoptic postprocessing
        >>> predicted_panoptic_map = processor.post_process_panoptic_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]["segmentation"]
        >>> f"👉 Panoptic Predictions Shape: {list(predicted_panoptic_map.shape)}"
        '👉 Panoptic Predictions Shape: [512, 683]'
        ```
        