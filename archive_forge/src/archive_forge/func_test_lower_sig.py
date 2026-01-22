import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
def test_lower_sig(capfd, hello_world_f77, monkeypatch):
    """Lowers cases in signature files by flag or when -h is present

    CLI :: --[no-]lower -h
    """
    foutl = get_io_paths(hello_world_f77, mname='test')
    ipath = foutl.finp
    capshi = re.compile('Block: HI')
    capslo = re.compile('Block: hi')
    monkeypatch.setattr(sys, 'argv', f'f2py {ipath} -h {foutl.pyf} -m test --overwrite-signature'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert capslo.search(out) is not None
        assert capshi.search(out) is None
    monkeypatch.setattr(sys, 'argv', f'f2py {ipath} -h {foutl.pyf} -m test --overwrite-signature --no-lower'.split())
    with util.switchdir(ipath.parent):
        f2pycli()
        out, _ = capfd.readouterr()
        assert capslo.search(out) is None
        assert capshi.search(out) is not None