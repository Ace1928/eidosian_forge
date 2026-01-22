import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_find_functiononly_keyerror():
    with pytest.raises(KeyError):
        rinterface.globalenv.find('pi', wantfun=True)