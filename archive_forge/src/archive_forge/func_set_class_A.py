import pytest
from rpy2 import robjects
@pytest.fixture(scope='module')
def set_class_A():
    robjects.r('methods::setClass("A", representation(a="numeric", b="character"))')
    yield
    robjects.r('methods::removeClass("A")')