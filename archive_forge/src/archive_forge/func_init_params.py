import logging
import copy
from ..initializer import Uniform
from .base_module import BaseModule
def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None, allow_missing=False, force_init=False, allow_extra=False):
    """Initializes parameters.

        Parameters
        ----------
        initializer : Initializer
        arg_params : dict
            Default ``None``. Existing parameters. This has higher priority
            than `initializer`.
        aux_params : dict
            Default ``None``. Existing auxiliary states. This has higher priority
            than `initializer`.
        allow_missing : bool
            Allow missing values in `arg_params` and `aux_params` (if not ``None``).
            In this case, missing values will be filled with `initializer`.
        force_init : bool
            Default ``False``.
        allow_extra : boolean, optional
            Whether allow extra parameters that are not needed by symbol.
            If this is True, no error will be thrown when arg_params or aux_params
            contain extra parameters that is not needed by the executor.
        """
    if self.params_initialized and (not force_init):
        return
    assert self.binded, 'call bind before initializing the parameters'
    for module in self._modules:
        module.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params, allow_missing=allow_missing, force_init=force_init, allow_extra=allow_extra)

    def _check_name(known_names, new_names, modules, i):
        """Internal function to help checking duplicated names."""
        for name in new_names:
            assert not name in known_names, 'Duplicated parameter names: ' + 'name "%s" in layer %d (%s) is already ' % (name, i, type(modules[i])) + 'used in layer %d (%s).' % (known_names[name], type(modules[known_names[name]]))
            known_names[name] = i
    arg_names = dict()
    aux_names = dict()
    for i_layer, module in enumerate(self._modules):
        arg_params, aux_params = module.get_params()
        _check_name(arg_names, arg_params.keys(), self._modules, i_layer)
        _check_name(aux_names, aux_params.keys(), self._modules, i_layer)
    self.params_initialized = True