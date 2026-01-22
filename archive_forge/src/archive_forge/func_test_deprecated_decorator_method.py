from __future__ import annotations
import inspect
import warnings
import pytest
from .._deprecate import (
from . import module_with_deprecations
def test_deprecated_decorator_method(recwarn_always: pytest.WarningsRecorder) -> None:
    f = Foo()
    assert f.method() == 7
    got = recwarn_always.pop(TrioDeprecationWarning)
    assert isinstance(got.message, Warning)
    assert 'test_deprecate.Foo.method is deprecated' in got.message.args[0]