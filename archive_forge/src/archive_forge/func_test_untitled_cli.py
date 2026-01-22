import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
@pytest.mark.skipif(platform.system() != 'Linux' or sys.version_info <= (3, 12), reason='Compiler and 3.12 required')
def test_untitled_cli(capfd, hello_world_f90, monkeypatch):
    """Check that modules are named correctly

    CLI :: defaults
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, 'argv', f'f2py --backend meson -c {ipath}'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert 'untitledmodule.c' in out