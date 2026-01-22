from __future__ import annotations
import io
import sys
from typing import TYPE_CHECKING
import pytest
from trio._tools.mypy_annotate import Result, export, main, process_line
@pytest.mark.parametrize(('src', 'expected'), [('', None), ('a regular line\n', None), ('package\\filename.py:42:8: note: Some info\n', Result(kind='notice', filename='package\\filename.py', start_line=42, start_col=8, end_line=None, end_col=None, message=' Some info')), ('package/filename.py:42:1:46:3: error: Type error here [code]\n', Result(kind='error', filename='package/filename.py', start_line=42, start_col=1, end_line=46, end_col=3, message=' Type error here [code]')), ('package/module.py:87: warn: Bad code\n', Result(kind='warning', filename='package/module.py', start_line=87, message=' Bad code'))], ids=['blank', 'normal', 'note-wcol', 'error-wend', 'warn-lineonly'])
def test_processing(src: str, expected: Result | None) -> None:
    result = process_line(src)
    assert result == expected