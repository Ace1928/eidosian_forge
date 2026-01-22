import copy
import gc
import inspect
import runpy
import sys
import threading
from collections import namedtuple
from enum import Enum
from functools import wraps, partial
from typing import List, Any, ClassVar, Optional, Sequence, Tuple, Union, Dict, Set
import unittest
import os
import torch
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM, TEST_MKL, \
from torch.testing._internal.common_cuda import _get_torch_cuda_version, \
from torch.testing._internal.common_dtype import get_all_dtypes
def instantiate_device_type_tests(generic_test_class, scope, except_for=None, only_for=None, include_lazy=False, allow_mps=False):
    del scope[generic_test_class.__name__]
    empty_name = generic_test_class.__name__ + '_base'
    empty_class = type(empty_name, generic_test_class.__bases__, {})
    generic_members = set(generic_test_class.__dict__.keys()) - set(empty_class.__dict__.keys())
    generic_tests = [x for x in generic_members if x.startswith('test')]
    test_bases = device_type_test_bases.copy()
    if allow_mps and TEST_MPS and (MPSTestBase not in test_bases):
        test_bases.append(MPSTestBase)
    desired_device_type_test_bases = filter_desired_device_types(test_bases, except_for, only_for)
    if include_lazy:
        if IS_FBCODE:
            print('TorchScript backend not yet supported in FBCODE/OVRSOURCE builds', file=sys.stderr)
        else:
            desired_device_type_test_bases.append(LazyTestBase)

    def split_if_not_empty(x: str):
        return x.split(',') if len(x) != 0 else []
    env_only_for = split_if_not_empty(os.getenv(PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, ''))
    env_except_for = split_if_not_empty(os.getenv(PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY, ''))
    desired_device_type_test_bases = filter_desired_device_types(desired_device_type_test_bases, env_except_for, env_only_for)
    for base in desired_device_type_test_bases:
        class_name = generic_test_class.__name__ + base.device_type.upper()
        device_type_test_class: Any = type(class_name, (base, empty_class), {})
        for name in generic_members:
            if name in generic_tests:
                test = getattr(generic_test_class, name)
                sig = inspect.signature(device_type_test_class.instantiate_test)
                if len(sig.parameters) == 3:
                    device_type_test_class.instantiate_test(name, copy.deepcopy(test), generic_cls=generic_test_class)
                else:
                    device_type_test_class.instantiate_test(name, copy.deepcopy(test))
            else:
                assert name not in device_type_test_class.__dict__, f'Redefinition of directly defined member {name}'
                nontest = getattr(generic_test_class, name)
                setattr(device_type_test_class, name, nontest)
        device_type_test_class.__module__ = generic_test_class.__module__
        scope[class_name] = device_type_test_class