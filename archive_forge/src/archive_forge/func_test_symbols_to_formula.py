import pytest
from ase import Atoms
from ase.build import molecule
from ase.symbols import Symbols
def test_symbols_to_formula():
    symstr = 'CH3CH2OH'
    symbols = Symbols.fromsymbols(symstr)
    assert str(symbols.formula) == symstr