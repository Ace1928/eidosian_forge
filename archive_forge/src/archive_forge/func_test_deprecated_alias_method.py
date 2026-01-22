from __future__ import annotations
import inspect
import warnings
import pytest
from .._deprecate import (
from . import module_with_deprecations
def test_deprecated_alias_method(recwarn_always: pytest.WarningsRecorder) -> None:
    obj = Alias()
    assert obj.old_hotness_method() == 'new hotness method'
    got = recwarn_always.pop(TrioDeprecationWarning)
    assert isinstance(got.message, Warning)
    msg = got.message.args[0]
    assert 'test_deprecate.Alias.old_hotness_method is deprecated' in msg
    assert 'test_deprecate.Alias.new_hotness_method instead' in msg