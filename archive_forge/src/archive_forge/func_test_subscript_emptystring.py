import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_subscript_emptystring():
    ge = rinterface.globalenv
    with pytest.raises(ValueError):
        ge['']