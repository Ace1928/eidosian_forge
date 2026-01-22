from IPython.utils.dir2 import dir2
import pytest
def test_SubClass():

    class SubClass(Base):
        y = 2
    res = dir2(SubClass())
    assert 'y' in res
    assert res.count('y') == 1
    assert res.count('x') == 1