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
def get_nodes_number_difference(onnx_model_path: Union[str, os.PathLike], onnx_optimized_model_path: Union[str, os.PathLike]) -> int:
    """
        Compute the difference in the number of nodes between the original and the optimized model.

        Args:
            onnx_model_path (`Union[str, os.PathLike]`):
                Path of the ONNX model.
            onnx_optimized_model_path (`Union[str, os.PathLike]`):
                Path of the optimized ONNX model.

        Returns:
            The difference in the number of nodes between the original and the optimized model.
        """
    onnx_model = BertOnnxModel(load_model(onnx_model_path))
    onnx_optimized_model = BertOnnxModel(load_model(onnx_optimized_model_path))
    nodes_number_onnx_model = len(onnx_model.nodes())
    nodes_number_onnx_optimized_model = len(onnx_optimized_model.nodes())
    difference_nodes_number = nodes_number_onnx_model - nodes_number_onnx_optimized_model
    logger.info(f'There are {nodes_number_onnx_model} nodes before optimization and {nodes_number_onnx_optimized_model}nodes after. The number of nodes removed is {difference_nodes_number}')
    return difference_nodes_number