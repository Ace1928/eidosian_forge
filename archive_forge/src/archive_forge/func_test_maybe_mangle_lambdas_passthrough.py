import numpy as np
import pytest
from pandas.core.apply import (
def test_maybe_mangle_lambdas_passthrough():
    assert maybe_mangle_lambdas('mean') == 'mean'
    assert maybe_mangle_lambdas(lambda x: x).__name__ == '<lambda>'
    assert maybe_mangle_lambdas([lambda x: x])[0].__name__ == '<lambda>'