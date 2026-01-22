import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_find_invalid_empty():
    with pytest.raises(ValueError):
        rinterface.globalenv.find('')