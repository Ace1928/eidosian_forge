import os
import pathlib
import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import tobytes
from pyarrow.lib import ArrowInvalid, ArrowNotImplementedError
def test_get_supported_functions():
    supported_functions = pa._substrait.get_supported_functions()
    assert has_function(supported_functions, 'functions_arithmetic.yaml', 'add')
    assert has_function(supported_functions, 'functions_arithmetic.yaml', 'sum')