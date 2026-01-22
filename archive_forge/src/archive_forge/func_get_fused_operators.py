import gc
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union
import onnx
from onnx import load_model
from transformers.models.auto.configuration_auto import AutoConfig
from onnxruntime.transformers.onnx_model_bert import BertOnnxModel
from onnxruntime.transformers.optimizer import optimize_model
from ..onnx.utils import check_model_uses_external_data
from ..utils import CONFIG_NAME, NormalizedConfigManager, logging
from ..utils.save_utils import maybe_save_preprocessors
from .configuration import OptimizationConfig, ORTConfig
from .modeling_decoder import ORTModelForCausalLM
from .modeling_ort import ORTModel
from .modeling_seq2seq import ORTModelForConditionalGeneration
from .utils import ONNX_WEIGHTS_NAME, ORTConfigManager
@staticmethod
def get_fused_operators(onnx_model_path: Union[str, os.PathLike]) -> Dict[str, int]:
    """
        Computes the dictionary mapping the name of the fused operators to their number of apparition in the model.

        Args:
            onnx_model_path (`Union[str, os.PathLike]`):
                Path of the ONNX model.

        Returns:
            The dictionary mapping the name of the fused operators to their number of apparition in the model.
        """
    onnx_optimized_model = BertOnnxModel(load_model(onnx_model_path))
    fused_operator = onnx_optimized_model.get_fused_operator_statistics()
    logger.info(f'The following operators were fused : {', '.join([k for k, v in fused_operator.items() if v > 0])}')
    return {k: v for k, v in fused_operator.items() if v > 0}