import contextlib
import functools
import inspect
import pydoc
from .. import functions as fn
from . import Parameter
from .parameterTypes import ActionGroupParameter
def removeParameters(self, clearCache=True):
    """
        Disconnects from all signals of parameters in ``self.parameters``. Also,
        optionally clears the old cache of param values
        """
    for p in self.parameters.values():
        self._disconnectParameter(p)
    self.parameters.clear()
    if clearCache:
        self.parameterCache.clear()