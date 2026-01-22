import logging
import warnings
from .. import context as ctx
from .. import optimizer as opt
from .. import ndarray as nd
from .executor_group import DataParallelExecutorGroup
from ..model import _create_kvstore, _initialize_kvstore, _update_params, _update_params_on_kvstore
from ..model import load_checkpoint
from ..initializer import Uniform, InitDesc
from ..io import DataDesc
from ..ndarray import zeros
from .base_module import BaseModule, _check_input_names, _parse_data_desc
def load_optimizer_states(self, fname):
    """Loads optimizer (updater) state from a file.

        Parameters
        ----------
        fname : str
            Path to input states file.
        """
    assert self.optimizer_initialized
    if self._update_on_kvstore:
        self._kvstore.load_optimizer_states(fname)
    else:
        self._updater.set_states(open(fname, 'rb').read())