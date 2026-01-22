import pytest
from ase.io.cif import CIFBlock, parse_loop, CIFLoop
def test_cifloop_roundtrip():
    loop = CIFLoop()
    loop.add('_potatoes', [2.5, 3.0, -1.0], '{:8.5f}')
    loop.add('_eggs', [1, 2, 3], '{:2d}')
    string = loop.tostring() + '\n'
    print('hmm', string)
    lines = string.splitlines()[::-1]
    assert lines.pop() == 'loop_'
    for line in lines:
        print(repr(line))
    dct = parse_loop(lines)
    assert dct['_potatoes'] == pytest.approx([2.5, 3.0, -1.0])
    assert dct['_eggs'] == [1, 2, 3]