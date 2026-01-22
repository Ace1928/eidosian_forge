import sys
from contextlib import contextmanager
from typing import (
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
@pytest.mark.parametrize('data,code,expected_items', [[SpecialTyping(), 'data.typed_dict()', {'year': int, 'name': str}]])
def test_mocks_items_of_call_results(data, code, expected_items):
    context = limited(data=data, HeapType=HeapType, StringAnnotation=StringAnnotation)
    result = guarded_eval(code, context)
    ipython_keys = result._ipython_key_completions_()
    for key, value in expected_items.items():
        assert isinstance(result[key], value)
        assert key in ipython_keys