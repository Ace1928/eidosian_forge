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
@staticmethod
def dataclass_to_dict(config) -> dict:
    new_config = {}
    if config is None:
        return new_config
    if isinstance(config, dict):
        return config
    for k, v in asdict(config).items():
        if isinstance(v, Enum):
            v = v.name
        elif isinstance(v, list):
            v = [elem.name if isinstance(elem, Enum) else elem for elem in v]
        new_config[k] = v
    return new_config