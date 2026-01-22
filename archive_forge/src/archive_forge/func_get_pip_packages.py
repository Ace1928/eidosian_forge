import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def get_pip_packages(run_lambda, patterns=None):
    """Return `pip list` output. Note: will also find conda-installed pytorch and numpy packages."""
    if patterns is None:
        patterns = DEFAULT_PIP_PATTERNS

    def run_with_pip(pip):
        out = run_and_read_all(run_lambda, pip + ['list', '--format=freeze'])
        return '\n'.join((line for line in out.splitlines() if any((name in line for name in patterns))))
    pip_version = 'pip3' if sys.version[0] == '3' else 'pip'
    out = run_with_pip([sys.executable, '-mpip'])
    return (pip_version, out)