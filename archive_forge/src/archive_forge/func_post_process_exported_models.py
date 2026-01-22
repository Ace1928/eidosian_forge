from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
from transformers.utils import is_tf_available
from ...onnx import merge_decoders
from ...utils import (
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .constants import ONNX_DECODER_MERGED_NAME, ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME
from .model_patcher import DecoderModelPatcher
def post_process_exported_models(self, path: Path, models_and_onnx_configs: Dict[str, Tuple[Union['PreTrainedModel', 'TFPreTrainedModel', 'ModelMixin'], 'OnnxConfig']], onnx_files_subpaths: List[str]):
    models_and_onnx_configs, onnx_files_subpaths = super().post_process_exported_models(path, models_and_onnx_configs, onnx_files_subpaths)
    if self.use_past is True and len(models_and_onnx_configs) == 3:
        models_and_onnx_configs[ONNX_DECODER_NAME][1]._decoder_onnx_config.is_merged = True
        models_and_onnx_configs[ONNX_DECODER_NAME][1]._decoder_onnx_config.use_cache_branch = False
        models_and_onnx_configs[ONNX_DECODER_NAME][1]._decoder_onnx_config.use_past_in_inputs = True
        models_and_onnx_configs[ONNX_DECODER_WITH_PAST_NAME][1]._decoder_onnx_config.use_cache_branch = True
        models_and_onnx_configs[ONNX_DECODER_WITH_PAST_NAME][1]._decoder_onnx_config.is_merged = True
    return (models_and_onnx_configs, onnx_files_subpaths)