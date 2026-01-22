from .. import utils
from .._lazyload import rpy2
from . import conversion
import functools
@property
def function(self):
    try:
        return self._function
    except AttributeError:
        self._function = self._build()
        return self._function