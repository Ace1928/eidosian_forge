import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_from_pyobject_invalid_tag():
    pyobject = 'ahaha'
    with pytest.raises(TypeError):
        rinterface.SexpExtPtr.from_pyobject(pyobject, tag=True)