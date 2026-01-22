from __future__ import annotations
import functools
import glob
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import unittest
from collections import defaultdict
from typing import Any, Callable, Iterable, Pattern, Sequence
from urllib.request import urlretrieve
import numpy as np
import onnx
import onnx.reference
from onnx import ONNX_ML, ModelProto, NodeProto, TypeProto, ValueInfoProto, numpy_helper
from onnx.backend.base import Backend
from onnx.backend.test.case.test_case import TestCase
from onnx.backend.test.loader import load_model_tests
from onnx.backend.test.runner.item import TestItem
def retry_execute(times: int) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    assert times >= 1

    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:

        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            for i in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    print(f'{i} times tried')
                    if i == times:
                        raise
                    time.sleep(5 * i)
        return wrapped
    return wrapper