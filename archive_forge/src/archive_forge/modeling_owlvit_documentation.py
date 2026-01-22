import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_owlvit import OwlViTConfig, OwlViTTextConfig, OwlViTVisionConfig

        Returns:

        Examples:
        ```python
        >>> import requests
        >>> from PIL import Image
        >>> import torch
        >>> from transformers import AutoProcessor, OwlViTForObjectDetection

        >>> processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
        >>> model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = [["a photo of a cat", "a photo of a dog"]]
        >>> inputs = processor(text=texts, images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        >>> target_sizes = torch.Tensor([image.size[::-1]])
        >>> # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores
        >>> results = processor.post_process_object_detection(
        ...     outputs=outputs, threshold=0.1, target_sizes=target_sizes
        ... )

        >>> i = 0  # Retrieve predictions for the first image for the corresponding text queries
        >>> text = texts[i]
        >>> boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        >>> for box, score, label in zip(boxes, scores, labels):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
        Detected a photo of a cat with confidence 0.707 at location [324.97, 20.44, 640.58, 373.29]
        Detected a photo of a cat with confidence 0.717 at location [1.46, 55.26, 315.55, 472.17]
        ```