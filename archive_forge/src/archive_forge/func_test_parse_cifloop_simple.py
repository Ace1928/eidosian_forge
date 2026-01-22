import pytest
from ase.io.cif import CIFBlock, parse_loop, CIFLoop
def test_parse_cifloop_simple():
    dct = parse_loop(['_apples', '_verysmallrocks', '2 200', '3 300', '4 400'][::-1])
    assert dct['_apples'] == [2, 3, 4]
    assert dct['_verysmallrocks'] == [200, 300, 400]