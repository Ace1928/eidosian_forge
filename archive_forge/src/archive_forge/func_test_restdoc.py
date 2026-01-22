import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
def test_restdoc(capfd, hello_world_f90, monkeypatch):
    """Ensures that RsT documentation is written out

    CLI :: --rest-doc
    """
    ipath = Path(hello_world_f90)
    mname = 'blah'
    monkeypatch.setattr(sys, 'argv', f'f2py -m {mname} {ipath} --rest-doc'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert 'ReST Documentation is saved to file' in out
        with Path(f'./{mname}module.rest').open() as orst:
            assert '.. -*- rest -*-' in orst.read()