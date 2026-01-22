import pytest
from ase.io.cif import CIFBlock, parse_loop, CIFLoop
def test_deuterium():
    symbols = ['H', 'D', 'D', 'He']
    block = CIFBlock('deuterium', dict(_atom_site_type_symbol=symbols))
    assert block.get_symbols() == ['H', 'H', 'H', 'He']
    masses = block._get_masses()
    assert all(masses.round().astype(int) == [1, 2, 2, 4])