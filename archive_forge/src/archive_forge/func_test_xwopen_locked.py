import pytest
from ase.utils import xwopen
def test_xwopen_locked():
    with xwopen(filename) as fd:
        assert fd is not None
        with xwopen(filename) as fd2:
            assert fd2 is None