from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms, FixedPlane
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
import re
import pytest
from ase.cli.template import prec_round, slice_split, \
from ase.io import read
def test_template_functions():
    """Test functions used in the template module."""
    num = 1.55749
    rnum = [prec_round(num, i) for i in range(1, 6)]
    assert rnum == [1.6, 1.56, 1.557, 1.5575, 1.55749]
    assert slice_split('a@1:3:1') == ('a', slice(1, 3, 1))
    sym = 'H'
    num = sym2num[sym]
    mf = MapFormatter().format
    sym2 = mf('{:h}', num)
    assert sym == sym2