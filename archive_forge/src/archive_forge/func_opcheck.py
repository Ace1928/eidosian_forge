import datetime
import difflib
import functools
import inspect
import json
import os
import re
import tempfile
import threading
import unittest
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch._dynamo
import torch.utils._pytree as pytree
from torch._dynamo.utils import clone_input
from torch._subclasses.schema_check_mode import SchemaCheckMode
from torch._utils_internal import get_file_path_2
from torch.overrides import TorchFunctionMode
from torch.testing._internal.optests import (
def opcheck(op: torch._ops.OperatorBase, args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]]=None, *, test_utils: Union[str, List[str]]='ALL', raise_exception: bool=True) -> Dict[str, str]:
    """Given an operator and some sample arguments, tests if the operator is
    registered correctly.

    We test the following (which are important for correctness in eager-mode
    PyTorch and with torch.compile):
    - test_schema: if the operator's schema is correct.
    - test_autograd_registration: if autograd was registered correctly,
        i.e. to the correct DispatchKey.
    - test_faketensor: If the operator has a FakeTensor implementation
        (and if it is correct).
    - test_aot_dispatch_static: If the operator works with
        AOTAutograd/AOTDispatch, which is one of the parts in the PT2 stack.
        Checks that the outputs (and gradients, if they are computable)
        of the operator are the same under eager-mode PyTorch and torch.compile.
    - test_aot_dispatch_dynamic: Same as aot_dispatch_static, but
        tests dynamic shapes instead of static shapes.

    For best results, please call ``opcheck`` multiple times with a
    representative set of inputs. For example, if your operator supports
    autograd, please use ``opcheck`` with inputs that require_grad.

    Args:
        op: The operator. Should look like torch.ops.aten.foo
        args: The args to the operator
        kwargs: The kwargs to the operator
        test_utils: Tests that we should run. Default: all of them.
            Example: ["test_schema", "test_faketensor"]
        raise_exception: If we should raise an exception on the first
            error. If False, we will return a dict with information
            on if each test passed or not.

    """
    if kwargs is None:
        kwargs = {}
    if isinstance(op, torch._ops.OpOverloadPacket):
        op = resolve_unique_overload_or_throw(op)
    if not isinstance(op, torch._ops.OpOverload):
        raise ValueError(f'opcheck(op, ...): op must be instance of torch._ops.OpOverload, e.g. torch.ops.aten.sin.default, got {type(op)}')
    if test_utils == 'ALL':
        test_utils = tuple(ALL_TEST_UTILS.keys())
    if isinstance(test_utils, str):
        test_utils = (test_utils,)
    if not isinstance(test_utils, (tuple, list)) or not set(test_utils).issubset(ALL_TEST_UTILS.keys()):
        raise ValueError(f'opcheck(op, ..., test_utils={test_utils}), expected test_utils to be subset of {tuple(ALL_TEST_UTILS.keys())} but it was not')
    results_dict = {}
    for test_util in test_utils:
        tester = ALL_TEST_UTILS[test_util]
        try:
            tester(op, args, kwargs)
            results_dict[test_util] = 'SUCCESS'
        except Exception as ex:
            if raise_exception:
                raise OpCheckError(f'opcheck(op, ...): {test_util} failed with {ex} (scroll up for stack trace)') from ex
            results_dict[test_util] = ex
    return results_dict