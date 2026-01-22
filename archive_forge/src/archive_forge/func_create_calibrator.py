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
def create_calibrator(self, onnx_model_path: Union[str, os.PathLike, Path], operators_to_quantize: Optional[List[str]], use_external_data_format: bool=False, force_symmetric_range: bool=False, augmented_model_name: str='augmented_model.onnx') -> CalibraterBase:
    kwargs = {'model': onnx_model_path, 'op_types_to_calibrate': operators_to_quantize or [], 'calibrate_method': self.method, 'augmented_model_path': augmented_model_name}
    if parse(ort_version) > Version('1.10.0'):
        kwargs['use_external_data_format'] = use_external_data_format
        kwargs['extra_options'] = {'symmetric': force_symmetric_range, 'num_bins': self.num_bins, 'num_quantized_bins': self.num_quantized_bins, 'percentile': self.percentile, 'moving_average': self.moving_average, 'averaging_constant': self.averaging_constant}
    return create_calibrator(**kwargs)