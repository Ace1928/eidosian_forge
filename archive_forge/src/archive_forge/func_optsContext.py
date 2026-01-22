import contextlib
import functools
import inspect
import pydoc
from .. import functions as fn
from . import Parameter
from .parameterTypes import ActionGroupParameter
@contextlib.contextmanager
def optsContext(self, **opts):
    """
        Creates a new context for ``opts``, where each is reset to the old value
        when the context expires

        Parameters
        ----------
        opts:
            Options to set, must be one of the keys in :attr:`_optNames`
        """
    oldOpts = self.setOpts(**opts)
    yield
    self.setOpts(**oldOpts)