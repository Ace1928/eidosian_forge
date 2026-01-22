import sys
import numpy as np
import pytest
@pytest.fixture
def max_digits():
    try:
        orig_max_str_digits = sys.get_int_max_str_digits()
        yield sys.set_int_max_str_digits
        sys.set_int_max_str_digits(orig_max_str_digits)
    except AttributeError:
        yield (lambda x: None)