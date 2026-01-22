import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
@pytest.mark.xfail(reason='Consistently fails on CI.')
def test_npdistop(hello_world_f90, monkeypatch):
    """
    CLI :: -c
    """
    ipath = Path(hello_world_f90)
    monkeypatch.setattr(sys, 'argv', f'f2py -m blah {ipath} -c'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        cmd_run = shlex.split('python -c "import blah; blah.hi()"')
        rout = subprocess.run(cmd_run, capture_output=True, encoding='UTF-8')
        eout = ' Hello World\n'
        assert rout.stdout == eout