import sys
from contextlib import contextmanager
from typing import (
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
@pytest.mark.parametrize('data,code,expected_attributes', [[SpecialTyping(), 'data.optional_float()', ['is_integer']], [SpecialTyping(), 'data.union_str_and_int()', ['capitalize', 'as_integer_ratio']], [SpecialTyping(), 'data.protocol()', ['test_method']], [SpecialTyping(), 'data.typed_dict()', ['keys', 'values', 'items']]])
def test_mocks_attributes_of_call_results(data, code, expected_attributes):
    context = limited(data=data, HeapType=HeapType, StringAnnotation=StringAnnotation)
    result = guarded_eval(code, context)
    for attr in expected_attributes:
        assert hasattr(result, attr)
        assert attr in dir(result)