from . import _bootstrap
import abc
import warnings
def module_repr(self, module):
    """Return a module's repr.

        Used by the module type when the method does not raise
        NotImplementedError.

        This method is deprecated.

        """
    warnings.warn('importlib.abc.Loader.module_repr() is deprecated and slated for removal in Python 3.12', DeprecationWarning)
    raise NotImplementedError