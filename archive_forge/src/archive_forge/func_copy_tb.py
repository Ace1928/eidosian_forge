from __future__ import annotations
from types import TracebackType
from typing import Any, ClassVar, cast
def copy_tb(base_tb: TracebackType, tb_next: TracebackType | None) -> TracebackType:
    try:
        raise ValueError
    except ValueError as exc:
        new_tb = exc.__traceback__
        assert new_tb is not None
    c_new_tb = CTraceback.from_address(id(new_tb))
    assert c_new_tb.tb_next is None
    if tb_next is not None:
        _ctypes.Py_INCREF(tb_next)
        c_new_tb.tb_next = id(tb_next)
    assert c_new_tb.tb_frame is not None
    _ctypes.Py_INCREF(base_tb.tb_frame)
    old_tb_frame = new_tb.tb_frame
    c_new_tb.tb_frame = id(base_tb.tb_frame)
    _ctypes.Py_DECREF(old_tb_frame)
    c_new_tb.tb_lasti = base_tb.tb_lasti
    c_new_tb.tb_lineno = base_tb.tb_lineno
    try:
        return new_tb
    finally:
        del new_tb, old_tb_frame