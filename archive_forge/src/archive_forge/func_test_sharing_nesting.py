import itertools
import weakref
from collections import Counter
import numpy as np
import pytest
from opt_einsum import (contract, contract_expression, contract_path, get_symbol, helpers, shared_intermediates)
from opt_einsum.backends import to_cupy, to_torch
from opt_einsum.contract import _einsum
from opt_einsum.parser import parse_einsum_input
from opt_einsum.sharing import (count_cached_ops, currently_sharing, get_sharing_cache)
@pytest.mark.parametrize('backend', backends)
def test_sharing_nesting(backend):
    eqs = ['ab,bc,cd->a', 'ab,bc,cd->b', 'ab,bc,cd->c', 'ab,bc,cd->c']
    views = helpers.build_views(eqs[0])
    shapes = [v.shape for v in views]
    refs = weakref.WeakValueDictionary()

    def method1(views):
        with shared_intermediates():
            w = contract_expression(eqs[0], *shapes)(*views, backend=backend)
            x = contract_expression(eqs[2], *shapes)(*views, backend=backend)
            result = contract_expression('a,b->', w.shape, x.shape)(w, x, backend=backend)
            refs['w'] = w
            refs['x'] = x
            del w, x
            assert 'w' in refs
            assert 'x' in refs
        assert 'w' not in refs, 'cache leakage'
        assert 'x' not in refs, 'cache leakage'
        return result

    def method2(views):
        with shared_intermediates():
            y = contract_expression(eqs[2], *shapes)(*views, backend=backend)
            z = contract_expression(eqs[3], *shapes)(*views, backend=backend)
            refs['y'] = y
            refs['z'] = z
            result = contract_expression('c,d->', y.shape, z.shape)(y, z, backend=backend)
            result = result + method1(views)
            del y, z
            assert 'y' in refs
            assert 'z' in refs
        assert 'y' not in refs
        assert 'z' not in refs
    method1(views)
    method2(views)