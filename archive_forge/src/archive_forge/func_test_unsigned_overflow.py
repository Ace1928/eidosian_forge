import pytest
from numpy.core._simd import targets
@pytest.mark.parametrize('sfx', unsigned_sfx)
def test_unsigned_overflow(self, sfx):
    nlanes = getattr(npyv, 'nlanes_' + sfx)
    maxu = (1 << int(sfx[1:])) - 1
    maxu_72 = (1 << 72) - 1
    lane = getattr(npyv, 'setall_' + sfx)(maxu_72)[0]
    assert lane == maxu
    lanes = getattr(npyv, 'load_' + sfx)([maxu_72] * nlanes)
    assert lanes == [maxu] * nlanes
    lane = getattr(npyv, 'setall_' + sfx)(-1)[0]
    assert lane == maxu
    lanes = getattr(npyv, 'load_' + sfx)([-1] * nlanes)
    assert lanes == [maxu] * nlanes