from .. import optimizer as opt
from ..model import _create_kvstore, _create_sparse_kvstore
from .parameter import ParameterDict, Parameter
from ..kvstore import KVStore
def save_states(self, fname):
    """Saves trainer states (e.g. optimizer, momentum) to a file.


        Parameters
        ----------
        fname : str
            Path to output states file.

        Note
        ----
        `optimizer.param_dict`, which contains Parameter information (such as
        `lr_mult` and `wd_mult`) will not be saved.
        """
    assert self._optimizer is not None
    if not self._kv_initialized:
        self._init_kvstore()
    if self._params_to_init:
        self._init_params()
    if self._update_on_kvstore:
        assert not self._params_to_init, 'Cannot save trainer states when some parameters are not yet initialized in kvstore.'
        self._kvstore.save_optimizer_states(fname, dump_optimizer=True)
    else:
        with open(fname, 'wb') as fout:
            fout.write(self._updaters[0].get_states(dump_optimizer=True))