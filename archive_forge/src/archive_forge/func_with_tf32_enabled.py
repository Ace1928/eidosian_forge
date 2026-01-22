import functools
import torch
import torch.cuda
from torch.testing._internal.common_utils import LazyVal, TEST_NUMBA, TEST_WITH_ROCM, TEST_CUDA, IS_WINDOWS
import inspect
import contextlib
def with_tf32_enabled(self, function_call):
    with tf32_on(self, tf32_precision):
        function_call()