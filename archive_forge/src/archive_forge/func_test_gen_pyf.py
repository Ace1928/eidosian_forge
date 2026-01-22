import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
def test_gen_pyf(capfd, hello_world_f90, monkeypatch):
    """Ensures that a signature file is generated via the CLI
    CLI :: -h
    """
    ipath = Path(hello_world_f90)
    opath = Path(hello_world_f90).stem + '.pyf'
    monkeypatch.setattr(sys, 'argv', f'f2py -h {opath} {ipath}'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert 'Saving signatures to file' in out
        assert Path(f'{opath}').exists()