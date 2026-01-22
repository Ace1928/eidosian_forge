from __future__ import annotations
import inspect
import warnings
import pytest
from .._deprecate import (
from . import module_with_deprecations
def test_warn_deprecated_formatting(recwarn_always: pytest.WarningsRecorder) -> None:
    warn_deprecated(old, '1.0', issue=1, instead=new)
    got = recwarn_always.pop(TrioDeprecationWarning)
    assert isinstance(got.message, Warning)
    assert 'test_deprecate.old is deprecated' in got.message.args[0]
    assert 'test_deprecate.new instead' in got.message.args[0]