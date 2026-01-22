import os
import pytest
import numpy as np
import ase
import ase.lattice.cubic
from ase.calculators.castep import (Castep, CastepOption,
def test_castep_cell(testing_keywords):
    ccell = CastepCell(testing_keywords, keyword_tolerance=2)
    ccell.species_pot = ('H', 'H_test.usp')
    assert ccell.species_pot.value == '\nH H_test.usp'
    ccell.species_pot = [('H', 'H_test.usp'), ('He', 'He_test.usp')]
    assert ccell.species_pot.value == '\nH H_test.usp\nHe He_test.usp'
    R = np.array([np.eye(3), -np.eye(3)])
    T = np.zeros((2, 3))
    ccell.symmetry_ops = (R, T)
    strblock = [l.strip() for l in ccell.symmetry_ops.value.split('\n') if l.strip() != '']
    fblock = np.array([list(map(float, l.split())) for l in strblock])
    assert np.isclose(fblock[:3], R[0]).all()
    assert np.isclose(fblock[3], T[0]).all()
    assert np.isclose(fblock[4:7], R[1]).all()
    assert np.isclose(fblock[7], T[1]).all()
    a = ase.Atoms('H', positions=[[0, 0, 1]], cell=np.eye(3) * 2)
    ccell.positions_abs_product = a
    ccell.positions_abs_intermediate = a

    def parse_posblock(pblock, has_units=False):
        lines = pblock.split('\n')
        units = None
        if has_units:
            units = lines.pop(0).strip()
        pos_lines = []
        while len(lines) > 0:
            l = lines.pop(0).strip()
            if l == '':
                continue
            el, x, y, z = l.split()
            xyz = np.array(list(map(float, [x, y, z])))
            pos_lines.append((el, xyz))
        return (units, pos_lines)
    pap = parse_posblock(ccell.positions_abs_product.value, True)
    pai = parse_posblock(ccell.positions_abs_intermediate.value, True)
    assert pap[0] == 'ang'
    assert pap[1][0][0] == 'H'
    assert np.isclose(pap[1][0][1], a.get_positions()[0]).all()
    assert pai[0] == 'ang'
    assert pai[1][0][0] == 'H'
    assert np.isclose(pai[1][0][1], a.get_positions()[0]).all()
    ccell.positions_frac_product = a
    ccell.positions_frac_intermediate = a
    pfp = parse_posblock(ccell.positions_frac_product.value)
    pfi = parse_posblock(ccell.positions_frac_intermediate.value)
    assert pfp[1][0][0] == 'H'
    assert np.isclose(pfp[1][0][1], a.get_scaled_positions()[0]).all()
    assert pfi[1][0][0] == 'H'
    assert np.isclose(pfi[1][0][1], a.get_scaled_positions()[0]).all()
    ccell.kpoint_mp_grid = '3 3 3'
    with pytest.warns(UserWarning):
        ccell.kpoint_mp_spacing = 10.0