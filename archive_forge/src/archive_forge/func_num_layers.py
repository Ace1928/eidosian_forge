from collections import OrderedDict
from typing import Any, List, Mapping, Optional
from ... import PreTrainedTokenizer, TensorType, is_torch_available
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast, PatchingSpec
from ...utils import logging
@property
def num_layers(self) -> int:
    return self._config.n_layer