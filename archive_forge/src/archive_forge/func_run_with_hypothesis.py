from __future__ import annotations
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from inspect import isasyncgenfunction, iscoroutinefunction
from typing import Any, Dict, Tuple, cast
import pytest
import sniffio
from ._core._eventloop import get_all_backends, get_async_backend
from .abc import TestRunner
def run_with_hypothesis(**kwargs: Any) -> None:
    with get_runner(backend_name, backend_options) as runner:
        runner.run_test(original_func, kwargs)