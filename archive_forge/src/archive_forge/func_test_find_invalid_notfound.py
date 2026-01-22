import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_find_invalid_notfound():
    with pytest.raises(KeyError):
        rinterface.globalenv.find('asdf')