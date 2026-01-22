import pytest
import numpy as np
from ase.io import write
@pytest.fixture
def calc_settings():
    """Some simple fast calculation settings"""
    return dict(xc='lda', prec='Low', algo='Fast', setups='minimal', ismear=0, nelm=1, sigma=1.0, istart=0, lwave=False, lcharg=False)