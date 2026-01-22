import pytest
from statsmodels.tools.docstring import Docstring, remove_parameters, Parameter
def test_replace_block():
    ds = Docstring(good)
    ds.replace_block('summary', ['The is the new summary.'])
    assert 'The is the new summary.' in str(ds)
    ds = Docstring(good)
    ds.replace_block('summary', 'The is the new summary.')
    assert 'The is the new summary.' in str(ds)
    with pytest.raises(ValueError):
        ds.replace_block('unknown', ['The is the new summary.'])