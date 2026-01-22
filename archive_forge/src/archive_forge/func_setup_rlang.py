from .. import utils
from .._lazyload import rpy2
from . import conversion
import functools
@functools.lru_cache(None)
def setup_rlang():
    rpy2.robjects.r("\n    if (!require('rlang')) install.packages('rlang')\n    options(error = rlang::entrace)\n    ")