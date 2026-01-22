import os
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datasets import Dataset
from packaging.version import Version, parse
from onnxruntime import __version__ as ort_version
from onnxruntime.quantization import CalibraterBase, CalibrationMethod, QuantFormat, QuantizationMode, QuantType
from onnxruntime.quantization.calibrate import create_calibrator
from onnxruntime.quantization.registry import IntegerOpsRegistry, QDQRegistry, QLinearOpsRegistry
from onnxruntime.transformers.fusion_options import FusionOptions
from ..configuration_utils import BaseConfig
from ..utils import logging
def default_quantization_parameters(is_static: bool, format: Optional[QuantFormat]=None, mode: Optional[QuantizationMode]=None, operators_to_quantize: Optional[List[str]]=None) -> Tuple[QuantFormat, QuantizationMode, List[str]]:
    if format is None:
        format = QuantFormat.QDQ if is_static else QuantFormat.QOperator
    if mode is None:
        mode = QuantizationMode.QLinearOps if is_static else QuantizationMode.IntegerOps
    if operators_to_quantize is None or len(operators_to_quantize) == 0:
        if is_static and format == QuantFormat.QDQ:
            operators_to_quantize = ORT_DEFAULT_OPS_STATIC_QUANTIZATION_QDQ
        elif is_static and mode == QuantizationMode.QLinearOps:
            operators_to_quantize = ORT_DEFAULT_OPS_STATIC_QUANTIZATION_QOPS
        elif not is_static and mode == QuantizationMode.IntegerOps:
            operators_to_quantize = ORT_DEFAULT_OPS_DYNAMIC_QUANTIZATION
    return (format, mode, operators_to_quantize)