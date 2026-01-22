from collections import defaultdict
import pytest
from modin.config import Parameter
@pytest.mark.parametrize('vartype', [bool, int, dict])
def test_init_validation(vartype):
    parameter = make_prefilled(vartype, 'bad value')
    with pytest.raises(ValueError):
        parameter.get()