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
def inputs_for_causal_lm(self):
    if self.use_past_in_inputs:
        common_inputs = {'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size', 1: 'past_sequence_length + 1'}}
        for i in range(self._normalized_config.decoder_num_layers):
            common_inputs[f'past_key_values.{i}.key'] = {0: 'batch_size', 2: 'past_sequence_length'}
            common_inputs[f'past_key_values.{i}.value'] = {0: 'batch_size', 2: 'past_sequence_length'}
    else:
        common_inputs = {'input_ids': {0: 'batch_size', 1: 'sequence_length'}, 'attention_mask': {0: 'batch_size', 1: 'sequence_length'}}
    return common_inputs