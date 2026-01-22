from __future__ import annotations
import inspect
import warnings
import pytest
from .._deprecate import (
from . import module_with_deprecations
def test_warn_deprecated(recwarn_always: pytest.WarningsRecorder) -> None:

    def deprecated_thing() -> None:
        warn_deprecated('ice', '1.2', issue=1, instead='water')
    deprecated_thing()
    filename, lineno = _here()
    assert len(recwarn_always) == 1
    got = recwarn_always.pop(TrioDeprecationWarning)
    assert isinstance(got.message, Warning)
    assert 'ice is deprecated' in got.message.args[0]
    assert 'Trio 1.2' in got.message.args[0]
    assert 'water instead' in got.message.args[0]
    assert '/issues/1' in got.message.args[0]
    assert got.filename == filename
    assert got.lineno == lineno - 1