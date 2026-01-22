from __future__ import annotations
import re
import typing
from typing import Any, Callable, TypeVar
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import _api
def test_make_keyword_only() -> None:

    @_api.make_keyword_only('3.0', 'arg')
    def func(pre: Any, arg: Any, post: Any=None) -> None:
        pass
    func(1, arg=2)
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        func(1, 2)
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        func(1, 2, 3)