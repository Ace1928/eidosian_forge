import pytest
from numpy.core._simd import targets
@pytest.mark.parametrize('sfx', all_sfx)
def test_num_lanes(self, sfx):
    nlanes = getattr(npyv, 'nlanes_' + sfx)
    vector = getattr(npyv, 'setall_' + sfx)(1)
    assert len(vector) == nlanes