from __future__ import annotations
import gc
import sys
from traceback import extract_tb
from typing import TYPE_CHECKING, Callable, NoReturn
import pytest
from .._concat_tb import concat_tb
def raiser2_2() -> NoReturn:
    raise KeyError('raiser2_string')