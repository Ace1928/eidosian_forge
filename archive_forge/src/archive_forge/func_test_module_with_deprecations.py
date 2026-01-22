from __future__ import annotations
import inspect
import warnings
import pytest
from .._deprecate import (
from . import module_with_deprecations
def test_module_with_deprecations(recwarn_always: pytest.WarningsRecorder) -> None:
    assert module_with_deprecations.regular == 'hi'
    assert len(recwarn_always) == 0
    filename, lineno = _here()
    assert module_with_deprecations.dep1 == 'value1'
    got = recwarn_always.pop(TrioDeprecationWarning)
    assert isinstance(got.message, Warning)
    assert got.filename == filename
    assert got.lineno == lineno + 1
    assert 'module_with_deprecations.dep1' in got.message.args[0]
    assert 'Trio 1.1' in got.message.args[0]
    assert '/issues/1' in got.message.args[0]
    assert 'value1 instead' in got.message.args[0]
    assert module_with_deprecations.dep2 == 'value2'
    got = recwarn_always.pop(TrioDeprecationWarning)
    assert isinstance(got.message, Warning)
    assert 'instead-string instead' in got.message.args[0]
    with pytest.raises(AttributeError):
        module_with_deprecations.asdf