from __future__ import annotations
import re
import typing
from typing import Any, Callable, TypeVar
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import _api
def pyplot_wrapper(foo: Any=_api.deprecation._deprecated_parameter) -> None:
    func1(foo)