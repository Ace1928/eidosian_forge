import operator
from numba.core import config, utils
from numba.core.targetconfig import TargetConfig, Option
def include_default_options(*args):
    """Returns a mixin class with a subset of the options

    Parameters
    ----------
    *args : str
        Option names to include.
    """
    glbs = {k: getattr(DefaultOptions, k) for k in args}
    return type('OptionMixins', (), glbs)