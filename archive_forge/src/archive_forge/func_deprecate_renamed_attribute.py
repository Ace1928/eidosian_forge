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
def deprecate_renamed_attribute(old_name, new_name, mapping_func=None):
    if getattr(self, old_name, None) is not None:
        if mapping_func is None:

            def identity(x):
                return x
            mapping_func = identity
        setattr(self, new_name, mapping_func(getattr(self, old_name)))
        warnings.warn(f'{old_name} will be deprecated soon, use {new_name} instead, {new_name} is set to {getattr(self, new_name)}.', FutureWarning)