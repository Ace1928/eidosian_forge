from collections import OrderedDict
from typing import Any, Mapping, Optional
from ... import PreTrainedTokenizer, TensorType, is_torch_available
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast
from ...utils import logging
@staticmethod
def expand_attention_types_params(attention_types):
    attentions = []
    for item in attention_types:
        for _ in range(item[1]):
            attentions.extend(item[0])
    return attentions