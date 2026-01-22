import os
import pytest
from ase.build import bulk
from .filecmp_ignore_whitespace import filecmp_ignore_whitespace
def check_kpoints_line(n, contents):
    """Assert the contents of a line"""
    with open('KPOINTS', 'r') as fd:
        lines = fd.readlines()
    assert lines[n].strip() == contents