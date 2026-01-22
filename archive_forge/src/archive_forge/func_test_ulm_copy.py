import pytest
import numpy as np
import ase.io.ulm as ulm
def test_ulm_copy(ulmfile):
    path = ulmfile.with_name('c.ulm')
    ulm.copy(ulmfile, path, exclude={'.a'})
    with ulm.open(path) as r:
        assert 'a' not in r
        assert 'y' in r