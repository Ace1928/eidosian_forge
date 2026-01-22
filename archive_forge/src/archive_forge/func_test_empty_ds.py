import pytest
from statsmodels.tools.docstring import Docstring, remove_parameters, Parameter
def test_empty_ds():
    ds = Docstring(None)
    ds.replace_block('summary', ['The is the new summary.'])
    ds.remove_parameters('x')
    new = Parameter('w', 'ndarray', ['An array input.'])
    ds.insert_parameters('y', new)
    assert str(ds) == 'None'