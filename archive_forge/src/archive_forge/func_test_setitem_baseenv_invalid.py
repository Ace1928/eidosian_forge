import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_setitem_baseenv_invalid():
    with pytest.raises(ValueError):
        rinterface.baseenv['pi'] = 42