import textwrap, re, sys, subprocess, shlex
from pathlib import Path
from collections import namedtuple
import platform
import pytest
from . import util
from numpy.f2py.f2py2e import main as f2pycli
@pytest.fixture(scope='session')
def retreal_f77(tmpdir_factory):
    """Generates a single f77 file for testing"""
    fdat = util.getpath('tests', 'src', 'return_real', 'foo77.f').read_text()
    fn = tmpdir_factory.getbasetemp() / 'foo.f'
    fn.write_text(fdat, encoding='ascii')
    return fn