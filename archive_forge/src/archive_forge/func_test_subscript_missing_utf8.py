import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_subscript_missing_utf8():
    env = rinterface.baseenv['new.env']()
    with pytest.raises(KeyError), pytest.warns(rinterface.RRuntimeWarning):
        env['呵呵']