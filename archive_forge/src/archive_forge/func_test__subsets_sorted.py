from __future__ import print_function
from patsy.util import no_pickling
def test__subsets_sorted():
    assert list(_subsets_sorted((1, 2))) == [(), (1,), (2,), (1, 2)]
    assert list(_subsets_sorted((1, 2, 3))) == [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    assert len(list(_subsets_sorted(range(5)))) == 2 ** 5