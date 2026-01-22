import numpy as np
import pytest
from opt_einsum import contract, contract_expression, contract_path, helpers
from opt_einsum.paths import linear_to_ssa, ssa_to_linear
@pytest.mark.parametrize('equation', tests)
def test_linear_vs_ssa(equation):
    views = helpers.build_views(equation)
    linear_path, _ = contract_path(equation, *views)
    ssa_path = linear_to_ssa(linear_path)
    linear_path2 = ssa_to_linear(ssa_path)
    assert linear_path2 == linear_path