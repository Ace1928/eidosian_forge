from __future__ import annotations  # isort: split
import __future__  # Regular import, not special!
import enum
import functools
import importlib
import inspect
import json
import socket as stdlib_socket
import sys
import types
from pathlib import Path, PurePath
from types import ModuleType
from typing import TYPE_CHECKING, Protocol
import attrs
import pytest
import trio
import trio.testing
from trio._tests.pytest_plugin import skip_if_optional_else_raise
from .. import _core, _util
from .._core._tests.tutil import slow
from .pytest_plugin import RUN_SLOW
def test_core_is_properly_reexported() -> None:
    sources = [trio, trio.lowlevel, trio.testing]
    for symbol in dir(_core):
        if symbol.startswith('_'):
            continue
        found = 0
        for source in sources:
            if symbol in dir(source) and getattr(source, symbol) is getattr(_core, symbol):
                found += 1
        print(symbol, found)
        assert found == 1