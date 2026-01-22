import contextlib
import functools
import inspect
import pydoc
from .. import functions as fn
from . import Parameter
from .parameterTypes import ActionGroupParameter
def updateCachedParameterValues(self, param, value):
    """
        This function is connected to ``sigChanged`` of every parameter associated with
        it. This way, those parameters don't have to be queried for their value every
        time InteractiveFunction is __call__'ed
        """
    self.parameterCache[param.name()] = value