import numpy as np
import pytest
from opt_einsum import contract, contract_path
def test_ellipse_input4():
    string = '...b,...a->...'
    views = build_views(string)
    ein = contract(string, *views, optimize=False)
    opt = contract(views[0], [Ellipsis, 1], views[1], [Ellipsis, 0], [Ellipsis])
    assert np.allclose(ein, opt)