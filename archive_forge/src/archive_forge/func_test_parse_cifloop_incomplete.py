import pytest
from ase.io.cif import CIFBlock, parse_loop, CIFLoop
def test_parse_cifloop_incomplete():
    with pytest.raises(RuntimeError):
        parse_loop(['_spam', '_eggs', '1 2', '1'][::-1])