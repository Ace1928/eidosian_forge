from collections import OrderedDict
from typing import Any, Mapping, Optional
from ... import PreTrainedTokenizer
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from ...onnx.utils import compute_effective_axis_dimension
from ...utils import TensorType, is_torch_available, logging
@property
def inputs(self) -> Mapping[str, Mapping[int, str]]:
    if self.task in ['default', 'seq2seq-lm']:
        common_inputs = OrderedDict([('input_ids', {0: 'batch', 1: 'encoder_sequence'}), ('attention_mask', {0: 'batch', 1: 'encoder_sequence'})])
        if self.use_past:
            common_inputs['decoder_input_ids'] = {0: 'batch'}
            common_inputs['decoder_attention_mask'] = {0: 'batch', 1: 'past_decoder_sequence + sequence'}
        else:
            common_inputs['decoder_input_ids'] = {0: 'batch', 1: 'decoder_sequence'}
            common_inputs['decoder_attention_mask'] = {0: 'batch', 1: 'decoder_sequence'}
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction='inputs')
    elif self.task == 'causal-lm':
        common_inputs = OrderedDict([('input_ids', {0: 'batch', 1: 'encoder_sequence'}), ('attention_mask', {0: 'batch', 1: 'encoder_sequence'})])
        if self.use_past:
            num_encoder_layers, _ = self.num_layers
            for i in range(num_encoder_layers):
                common_inputs[f'past_key_values.{i}.key'] = {0: 'batch', 2: 'past_sequence + sequence'}
                common_inputs[f'past_key_values.{i}.value'] = {0: 'batch', 2: 'past_sequence + sequence'}
    else:
        common_inputs = OrderedDict([('input_ids', {0: 'batch', 1: 'encoder_sequence'}), ('attention_mask', {0: 'batch', 1: 'encoder_sequence'}), ('decoder_input_ids', {0: 'batch', 1: 'decoder_sequence'}), ('decoder_attention_mask', {0: 'batch', 1: 'decoder_sequence'})])
    return common_inputs