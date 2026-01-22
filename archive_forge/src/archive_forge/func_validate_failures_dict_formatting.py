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
def validate_failures_dict_formatting(failures_dict_path: str) -> None:
    with open(failures_dict_path) as fp:
        actual = fp.read()
    failures_dict = FailuresDict.load(failures_dict_path)
    expected = failures_dict._save(to_str=True)
    if actual == expected:
        return
    if should_update_failures_dict():
        failures_dict = FailuresDict.load(failures_dict_path)
        failures_dict.save()
        return
    expected = expected.splitlines(1)
    actual = actual.splitlines(1)
    diff = difflib.unified_diff(actual, expected)
    diff = ''.join(diff)
    raise RuntimeError(f'\n{diff}\n\nExpected the failures dict to be formatted a certain way. Please see the above diff; you can correct this either manually or by re-running the test with PYTORCH_OPCHECK_ACCEPT=1')