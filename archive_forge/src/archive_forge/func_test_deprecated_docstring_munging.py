from __future__ import annotations
import inspect
import warnings
import pytest
from .._deprecate import (
from . import module_with_deprecations
def test_deprecated_docstring_munging() -> None:
    assert docstring_test1.__doc__ == 'Hello!\n\n.. deprecated:: 2.1\n   Use hi instead.\n   For details, see `issue #1 <https://github.com/python-trio/trio/issues/1>`__.\n\n'
    assert docstring_test2.__doc__ == 'Hello!\n\n.. deprecated:: 2.1\n   Use hi instead.\n\n'
    assert docstring_test3.__doc__ == 'Hello!\n\n.. deprecated:: 2.1\n   For details, see `issue #1 <https://github.com/python-trio/trio/issues/1>`__.\n\n'
    assert docstring_test4.__doc__ == 'Hello!\n\n.. deprecated:: 2.1\n\n'