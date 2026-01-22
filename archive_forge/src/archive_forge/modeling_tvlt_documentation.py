import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_tvlt import TvltConfig

        labels (`torch.LongTensor` of shape `(batch_size, num_labels)`, *optional*):
            Labels for computing the audiovisual loss. Indices should be in `[0, ..., num_classes-1]` where num_classes
            refers to the number of classes in audiovisual tasks.

        Return:

        Examples:
        ```python
        >>> from transformers import TvltProcessor, TvltForAudioVisualClassification
        >>> import numpy as np
        >>> import torch

        >>> num_frames = 8
        >>> images = list(np.random.randn(num_frames, 3, 224, 224))
        >>> audio = list(np.random.randn(10000))
        >>> processor = TvltProcessor.from_pretrained("ZinengTang/tvlt-base")
        >>> model = TvltForAudioVisualClassification.from_pretrained("ZinengTang/tvlt-base")
        >>> input_dict = processor(images, audio, sampling_rate=44100, return_tensors="pt")

        >>> outputs = model(**input_dict)
        >>> loss = outputs.loss
        ```