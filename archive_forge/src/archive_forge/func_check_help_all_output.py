from __future__ import annotations
import sys
from subprocess import PIPE, Popen
from typing import Any, Sequence
def check_help_all_output(pkg: str, subcommand: Sequence[str] | None=None) -> tuple[str, str]:
    """test that `python -m PKG --help-all` works"""
    cmd = [sys.executable, '-m', pkg]
    if subcommand:
        cmd.extend(subcommand)
    cmd.append('--help-all')
    out, err, rc = get_output_error_code(cmd)
    assert rc == 0, err
    assert 'Traceback' not in err
    assert 'Options' in out
    assert 'Class options' in out
    return (out, err)