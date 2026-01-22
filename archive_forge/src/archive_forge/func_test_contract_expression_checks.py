import numpy as np
import pytest
from opt_einsum import contract, contract_expression
def test_contract_expression_checks():
    with pytest.raises(ValueError):
        contract_expression('ab,bc->ac', (2, 3), (3, 4), optimize=False)
    with pytest.raises(ValueError):
        contract_expression('ab,bc->ac', (2, 3), (3, 4), (42, 42))
    out = np.empty((2, 4))
    with pytest.raises(ValueError):
        contract_expression('ab,bc->ac', (2, 3), (3, 4), out=out)
    expr = contract_expression('ab,bc->ac', (2, 3), (3, 4))
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 3))
    assert '`ContractExpression` takes exactly 2' in str(err.value)
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 3), np.random.rand(2, 3), np.random.rand(2, 3))
    assert '`ContractExpression` takes exactly 2' in str(err.value)
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 3, 4), np.random.rand(3, 4))
    assert 'Internal error while evaluating `ContractExpression`' in str(err.value)
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 4), np.random.rand(3, 4, 5))
    assert 'Internal error while evaluating `ContractExpression`' in str(err.value)
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 3), np.random.rand(3, 4), out=np.random.rand(2, 4, 6))
    assert 'Internal error while evaluating `ContractExpression`' in str(err.value)
    with pytest.raises(ValueError) as err:
        expr(np.random.rand(2, 3), np.random.rand(3, 4), order='F')
    assert 'only valid keyword arguments to a `ContractExpression`' in str(err.value)