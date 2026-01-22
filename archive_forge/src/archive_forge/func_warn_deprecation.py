import functools
import inspect
import warnings
def warn_deprecation(text):
    warnings.warn(text, category=DeprecationWarning, stacklevel=2)