import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_del_baseerror():
    with pytest.raises(ValueError):
        rinterface.baseenv.__delitem__('letters')