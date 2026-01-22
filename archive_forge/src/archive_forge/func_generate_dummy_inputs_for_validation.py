from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
from transformers.utils import is_tf_available
from ...onnx import merge_decoders
from ...utils import (
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .constants import ONNX_DECODER_MERGED_NAME, ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME
from .model_patcher import DecoderModelPatcher
def generate_dummy_inputs_for_validation(self, reference_model_inputs: Dict[str, Any], onnx_input_names: Optional[List[str]]=None) -> Dict[str, Any]:
    if self._behavior is ConfigBehavior.ENCODER:
        return self._encoder_onnx_config.generate_dummy_inputs_for_validation(reference_model_inputs)
    else:
        if self._behavior is ConfigBehavior.DECODER:
            reference_model_inputs['input_ids'] = reference_model_inputs.pop('decoder_input_ids')
        if 'encoder_outputs' in reference_model_inputs:
            if 'encoder_hidden_states' in onnx_input_names:
                reference_model_inputs['encoder_hidden_states'] = reference_model_inputs.pop('encoder_outputs')[0]
            else:
                reference_model_inputs.pop('encoder_outputs')
        return self._decoder_onnx_config.generate_dummy_inputs_for_validation(reference_model_inputs)