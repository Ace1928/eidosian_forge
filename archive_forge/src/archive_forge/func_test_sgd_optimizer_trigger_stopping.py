import numpy as np
from sklearn.neural_network._stochastic_optimizers import (
from sklearn.utils._testing import assert_array_equal
def test_sgd_optimizer_trigger_stopping():
    params = [np.zeros(shape) for shape in shapes]
    lr = 2e-06
    optimizer = SGDOptimizer(params, lr, lr_schedule='adaptive')
    assert not optimizer.trigger_stopping('', False)
    assert lr / 5 == optimizer.learning_rate
    assert optimizer.trigger_stopping('', False)