import time
import logging
import mxnet as mx
from mxnet.module import Module
from .svrg_optimizer import _SVRGOptimizer
def init_optimizer(self, kvstore='local', optimizer='sgd', optimizer_params=(('learning_rate', 0.01),), force_init=False):
    """Installs and initializes SVRGOptimizer. The SVRGOptimizer is a wrapper class for a regular optimizer that is
        passed in and a special AssignmentOptimizer to accumulate the full gradients.  If KVStore is 'local' or None,
        the full gradients will be accumulated locally without pushing to the KVStore. Otherwise, additional keys will
        be pushed to accumulate the full gradients in the KVStore.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The default value is not a dictionary,
            just to avoid pylint warning of dangerous default values.
        force_init : bool
            Default ``False``, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """
    self._param_dict = [{key: mx.nd.zeros(shape=value.shape, ctx=self._context[i]) for key, value in self.get_params()[0].items()} for i in range(self._ctx_len)]
    svrg_optimizer = self._create_optimizer(_SVRGOptimizer.__name__, default_opt=optimizer, kvstore=kvstore, optimizer_params=optimizer_params)
    super(SVRGModule, self).init_optimizer(kvstore=kvstore, optimizer=svrg_optimizer, optimizer_params=optimizer_params, force_init=force_init)
    if self._kvstore:
        for idx, param_on_devs in enumerate(self._exec_group.param_arrays):
            name = self._exec_group.param_names[idx]
            self._kvstore.init(name + '_full', mx.nd.zeros(shape=self._arg_params[name].shape))
            if self._update_on_kvstore:
                self._kvstore.pull(name + '_full', param_on_devs, priority=-idx)