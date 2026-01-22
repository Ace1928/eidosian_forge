import pytest
from pandas.errors import (
import pandas as pd
@pytest.mark.parametrize('is_local', [True, False])
def test_catch_undefined_variable_error(is_local):
    variable_name = 'x'
    if is_local:
        msg = f"local variable '{variable_name}' is not defined"
    else:
        msg = f"name '{variable_name}' is not defined"
    with pytest.raises(UndefinedVariableError, match=msg):
        raise UndefinedVariableError(variable_name, is_local)