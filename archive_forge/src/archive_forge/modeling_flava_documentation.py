import collections
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_flava import (

        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import FlavaForPreTraining, AutoProcessor

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> model = FlavaForPreTraining.from_pretrained("facebook/flava-full")
        >>> processor = AutoProcessor.from_pretrained("facebook/flava-full")

        >>> text = ["a photo of a cat"]

        >>> inputs = processor(
        ...     images=[image],
        ...     text=text,
        ...     return_masks=True,
        ...     return_codebook_pixels=True,
        ...     padding=True,
        ...     max_length=77,
        ...     return_tensors="pt",
        ... )


        >>> output = model(**inputs)
        ```

        Return:

        