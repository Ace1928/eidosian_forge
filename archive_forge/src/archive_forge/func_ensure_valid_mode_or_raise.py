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
def ensure_valid_mode_or_raise(use_static_quantization: bool, mode: QuantizationMode):
    if not use_static_quantization and mode == QuantizationMode.QLinearOps:
        raise ValueError('Invalid combination of use_static_quantization = False and mode = QuantizationMode.QLinearOps. OnnxRuntime dynamic quantization requires mode = QuantizationMode.IntegerOps')