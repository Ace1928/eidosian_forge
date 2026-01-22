import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
@property
def inputs_name(self):
    if self.is_generating_dummy_inputs:
        if self.task in ['fill-mask', 'text-classification']:
            return 'input_ids'
        else:
            return 'pixel_values'
    else:
        return 'inputs'