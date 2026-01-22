import importlib
from typing import TYPE_CHECKING, Optional
from packaging import version
from .base import HfQuantizer
from ..utils import is_auto_gptq_available, is_optimum_available, is_torch_available, logging
from ..utils.quantization_config import GPTQConfig, QuantizationConfigMixin

    Quantizer of the GPTQ method - for GPTQ the quantizer support calibration of the model through
    `auto_gptq` package. Quantization is done under the hood for users if they load a non-prequantized model.
    