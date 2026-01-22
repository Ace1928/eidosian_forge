from __future__ import annotations
import sys
from subprocess import PIPE, Popen
from typing import Any, Sequence
def check_help_output(pkg: str, subcommand: Sequence[str] | None=None) -> tuple[str, str]:
    """test that `python -m PKG [subcommand] -h` works"""
    cmd = [sys.executable, '-m', pkg]
    if subcommand:
        cmd.extend(subcommand)
    cmd.append('-h')
    out, err, rc = get_output_error_code(cmd)
    assert rc == 0, err
    assert 'Traceback' not in err
    assert 'Options' in out
    assert '--help-all' in out
    return (out, err)