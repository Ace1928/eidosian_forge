import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.fixture(autouse=True)
def skip_if_immutable(self, dtype, request):
    if dtype._is_immutable:
        node = request.node
        if node.name.split('[')[0] == 'test_is_immutable':
            return
        defined_in = node.function.__qualname__.split('.')[0]
        if defined_in == 'BaseSetitemTests':
            pytest.skip('__setitem__ test not applicable with immutable dtype')