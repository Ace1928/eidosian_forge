import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, is_torchvision_available, logging, requires_backends
from ...utils.backbone_utils import load_backbone
from .configuration_deta import DetaConfig
def load_cuda_kernels():
    from torch.utils.cpp_extension import load
    global MultiScaleDeformableAttention
    root = Path(__file__).resolve().parent.parent.parent / 'kernels' / 'deta'
    src_files = [root / filename for filename in ['vision.cpp', os.path.join('cpu', 'ms_deform_attn_cpu.cpp'), os.path.join('cuda', 'ms_deform_attn_cuda.cu')]]
    load('MultiScaleDeformableAttention', src_files, with_cuda=True, extra_include_paths=[str(root)], extra_cflags=['-DWITH_CUDA=1'], extra_cuda_cflags=['-DCUDA_HAS_FP16=1', '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__'])