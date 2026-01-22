import os
import pytest
import numpy as np
import ase
import ase.lattice.cubic
from ase.calculators.castep import (Castep, CastepOption,
def test_fundamental_params():
    boolOpt = CastepOption('test_bool', 'basic', 'defined')
    boolOpt.value = 'TRUE'
    assert boolOpt.raw_value is True
    float3Opt = CastepOption('test_float3', 'basic', 'real vector')
    float3Opt.value = '1.0 2.0 3.0'
    assert np.isclose(float3Opt.raw_value, [1, 2, 3]).all()
    mock_castep_keywords = CastepKeywords(make_param_dict(), make_cell_dict(), [], [], 0)
    mock_cparam = CastepParam(mock_castep_keywords, keyword_tolerance=2)
    mock_ccell = CastepCell(mock_castep_keywords, keyword_tolerance=2)
    mock_cparam.continuation = 'default'
    with pytest.warns(None):
        mock_cparam.reuse = 'default'
    assert mock_cparam.reuse.value is None
    mock_ccell.species_pot = ('Si', 'Si.usp')
    mock_ccell.species_pot = ('C', 'C.usp')
    assert 'Si Si.usp' in mock_ccell.species_pot.value
    assert 'C C.usp' in mock_ccell.species_pot.value
    symops = (np.eye(3)[None], np.zeros(3)[None])
    mock_ccell.symmetry_ops = symops
    assert '1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\n0.0 0.0 0.0' == mock_ccell.symmetry_ops.value.strip()