import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import prune_linear_layer
from ...utils import logging
from ...utils.backbone_utils import load_backbone
from .configuration_tvp import TvpConfig

        labels (`torch.FloatTensor` of shape `(batch_size, 3)`, *optional*):
            The labels contains duration, start time, and end time of the video corresponding to the text.
        Returns:

        Examples:
        ```python
        >>> import torch
        >>> from transformers import AutoConfig, AutoTokenizer, TvpForVideoGrounding

        >>> model = TvpForVideoGrounding.from_pretrained("Jiqing/tiny-random-tvp")

        >>> tokenizer = AutoTokenizer.from_pretrained("Jiqing/tiny-random-tvp")

        >>> pixel_values = torch.rand(1, 1, 3, 448, 448)
        >>> text_inputs = tokenizer("This is an example input", return_tensors="pt")
        >>> output = model(text_inputs.input_ids, pixel_values, text_inputs.attention_mask)
        ```