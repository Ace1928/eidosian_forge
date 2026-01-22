import numpy as np
import pytest
from opt_einsum import contract, contract_expression, contract_path, helpers
from opt_einsum.paths import linear_to_ssa, ssa_to_linear
@pytest.mark.parametrize('string,constants', [('hbc,bdef,cdkj,ji,ikeh,lfo', [1, 2, 3, 4]), ('bdef,cdkj,ji,ikeh,hbc,lfo', [0, 1, 2, 3]), ('hbc,bdef,cdkj,ji,ikeh,lfo', [1, 2, 3, 4]), ('hbc,bdef,cdkj,ji,ikeh,lfo', [1, 2, 3, 4]), ('ijab,acd,bce,df,ef->ji', [1, 2, 3, 4]), ('ab,cd,ad,cb', [1, 3]), ('ab,bc,cd', [0, 1])])
def test_contract_expression_with_constants(string, constants):
    views = helpers.build_views(string)
    expected = contract(string, *views, optimize=False, use_blas=False)
    shapes = [view.shape for view in views]
    expr_args = []
    ctrc_args = []
    for i, (shape, view) in enumerate(zip(shapes, views)):
        if i in constants:
            expr_args.append(view)
        else:
            expr_args.append(shape)
            ctrc_args.append(view)
    expr = contract_expression(string, *expr_args, constants=constants)
    print(expr)
    out = expr(*ctrc_args)
    assert np.allclose(expected, out)