from functools import partial
from typing import Any, List, Optional, Union
import torch
from torch import nn, Tensor
from torch.ao.quantization import DeQuantStub, QuantStub
from ...ops.misc import Conv2dNormActivation, SqueezeExcitation
from ...transforms._presets import ImageClassification
from .._api import register_model, Weights, WeightsEnum
from .._meta import _IMAGENET_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..mobilenetv3 import (
from .utils import _fuse_modules, _replace_relu

        MobileNet V3 main class

        Args:
           Inherits args from floating point MobileNetV3
        