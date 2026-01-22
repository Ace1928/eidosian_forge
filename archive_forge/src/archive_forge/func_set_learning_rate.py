from .. import optimizer as opt
from ..model import _create_kvstore, _create_sparse_kvstore
from .parameter import ParameterDict, Parameter
from ..kvstore import KVStore
def set_learning_rate(self, lr):
    """Sets a new learning rate of the optimizer.

        Parameters
        ----------
        lr : float
            The new learning rate of the optimizer.
        """
    if not isinstance(self._optimizer, opt.Optimizer):
        raise UserWarning('Optimizer has to be defined before its learning rate is mutated.')
    self._optimizer.set_learning_rate(lr)