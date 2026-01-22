import pytest
from rpy2 import robjects
def test_slotnames(set_class_A):
    ainstance = robjects.r('new("A", a=1, b="c")')
    assert tuple(ainstance.slotnames()) == ('a', 'b')