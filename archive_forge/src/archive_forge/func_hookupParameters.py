import contextlib
import functools
import inspect
import pydoc
from .. import functions as fn
from . import Parameter
from .parameterTypes import ActionGroupParameter
def hookupParameters(self, params=None, clearOld=True):
    """
        Binds a new set of parameters to this function. If ``clearOld`` is *True* (
        default), previously bound parameters are disconnected.

        Parameters
        ----------
        params: Sequence[Parameter]
            New parameters to listen for updates and optionally propagate keywords
            passed to :meth:`__call__`
        clearOld: bool
            If ``True``, previously hooked up parameters will be removed first
        """
    if clearOld:
        self.removeParameters()
    for param in params:
        self.parameters[param.name()] = param
        param.sigValueChanged.connect(self.updateCachedParameterValues)
        self.parameterCache[param.name()] = param.value() if param.hasValue() else None