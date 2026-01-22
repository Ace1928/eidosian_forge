import numpy as np
import pytest
from pandas import (
from pandas.core.strings.accessor import StringMethods
@pytest.mark.parametrize('dtype', [object, 'category'])
def test_api_per_method(index_or_series, dtype, any_allowed_skipna_inferred_dtype, any_string_method, request):
    box = index_or_series
    inferred_dtype, values = any_allowed_skipna_inferred_dtype
    method_name, args, kwargs = any_string_method
    reason = None
    if box is Index and values.size == 0:
        if method_name in ['partition', 'rpartition'] and kwargs.get('expand', True):
            raises = TypeError
            reason = 'Method cannot deal with empty Index'
        elif method_name == 'split' and kwargs.get('expand', None):
            raises = TypeError
            reason = 'Split fails on empty Series when expand=True'
        elif method_name == 'get_dummies':
            raises = ValueError
            reason = 'Need to fortify get_dummies corner cases'
    elif box is Index and inferred_dtype == 'empty' and (dtype == object) and (method_name == 'get_dummies'):
        raises = ValueError
        reason = 'Need to fortify get_dummies corner cases'
    if reason is not None:
        mark = pytest.mark.xfail(raises=raises, reason=reason)
        request.applymarker(mark)
    t = box(values, dtype=dtype)
    method = getattr(t.str, method_name)
    bytes_allowed = method_name in ['decode', 'get', 'len', 'slice']
    mixed_allowed = method_name not in ['cat']
    allowed_types = ['string', 'unicode', 'empty'] + ['bytes'] * bytes_allowed + ['mixed', 'mixed-integer'] * mixed_allowed
    if inferred_dtype in allowed_types:
        with option_context('future.no_silent_downcasting', True):
            method(*args, **kwargs)
    else:
        msg = f'Cannot use .str.{method_name} with values of inferred dtype {repr(inferred_dtype)}.'
        with pytest.raises(TypeError, match=msg):
            method(*args, **kwargs)