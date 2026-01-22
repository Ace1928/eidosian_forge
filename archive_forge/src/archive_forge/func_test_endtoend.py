from __future__ import annotations
import io
import sys
from typing import TYPE_CHECKING
import pytest
from trio._tools.mypy_annotate import Result, export, main, process_line
def test_endtoend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    inp_text = 'Mypy begun\ntrio/core.py:15: error: Bad types here [misc]\ntrio/package/module.py:48:4:56:18: warn: Missing annotations  [no-untyped-def]\nFound 3 errors in 29 files\n'
    result_file = tmp_path / 'dump.dat'
    assert not result_file.exists()
    with monkeypatch.context():
        monkeypatch.setattr(sys, 'stdin', io.StringIO(inp_text))
        main(['--dumpfile', str(result_file), '--platform', 'SomePlatform'])
    std = capsys.readouterr()
    assert std.err == ''
    assert std.out == inp_text
    assert result_file.exists()
    main(['--dumpfile', str(result_file)])
    std = capsys.readouterr()
    assert std.err == ''
    assert std.out == '::error file=trio/core.py,line=15,title=Mypy-SomePlatform::trio/core.py:15: Bad types here [misc]\n::warning file=trio/package/module.py,line=48,col=4,endLine=56,endColumn=18,title=Mypy-SomePlatform::trio/package/module.py:(48:4 - 56:18): Missing annotations  [no-untyped-def]\n'