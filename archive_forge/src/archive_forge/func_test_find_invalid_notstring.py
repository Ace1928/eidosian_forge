import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_find_invalid_notstring():
    with pytest.raises(TypeError):
        rinterface.globalenv.find(None)