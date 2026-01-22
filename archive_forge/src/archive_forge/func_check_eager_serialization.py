import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.nn.intrinsic import _FusedModule
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM
from torch.ao.quantization import (
from torch.ao.quantization import QuantWrapper, QuantStub, DeQuantStub, \
from torch.ao.quantization.quantization_mappings import (
from torch.testing._internal.common_quantized import (
from torch.jit.mobile import _load_for_lite_interpreter
import copy
import io
import functools
import time
import os
import unittest
import numpy as np
from torch.testing import FileCheck
from typing import Callable, Tuple, Dict, Any, Union, Type, Optional
import torch._dynamo as torchdynamo
def check_eager_serialization(self, ref_model, loaded_model, x):
    model_dict = ref_model.state_dict()
    b = io.BytesIO()
    torch.save(model_dict, b)
    b.seek(0)
    loaded_dict = torch.load(b)
    loaded_model.load_state_dict(loaded_dict)
    ref_out = ref_model(*x)
    load_out = loaded_model(*x)

    def check_outputs(ref_out, load_out):
        self.assertEqual(ref_out[0], load_out[0])
        if isinstance(ref_out[1], tuple):
            self.assertEqual(ref_out[1][0], load_out[1][0])
            self.assertEqual(ref_out[1][1], load_out[1][1])
        else:
            self.assertEqual(ref_out[1], load_out[1])
    check_outputs(ref_out, load_out)
    b = io.BytesIO()
    torch.save(ref_model, b)
    b.seek(0)
    loaded = torch.load(b)
    load_out = loaded(*x)
    check_outputs(ref_out, load_out)