from __future__ import annotations
import re
import typing
from typing import Any, Callable, TypeVar
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import _api
def test_empty_check_in_list() -> None:
    with pytest.raises(TypeError, match='No argument to check!'):
        _api.check_in_list(['a'])