import pytest
import rpy2.robjects as robjects
import rpy2.robjects.language as lg
from rpy2 import rinterface
from rpy2.rinterface_lib import embedded
@pytest.fixture(scope='module')
def clean_globalenv():
    yield
    for name in robjects.globalenv.keys():
        del robjects.globalenv[name]