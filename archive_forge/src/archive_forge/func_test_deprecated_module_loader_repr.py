import collections
import dataclasses
import importlib.metadata
import inspect
import logging
import multiprocessing
import os
import sys
import traceback
import types
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Tuple
from importlib.machinery import ModuleSpec
from unittest import mock
import duet
import numpy as np
import pandas as pd
import pytest
import sympy
from _pytest.outcomes import Failed
import cirq.testing
from cirq._compat import (
def test_deprecated_module_loader_repr():

    class StubLoader(importlib.abc.Loader):

        def module_repr(self, module: ModuleType) -> str:
            return 'hello'
    module = types.ModuleType('old')
    assert DeprecatedModuleLoader(StubLoader(), 'old_hello', 'new_hello').module_repr(module) == 'hello'