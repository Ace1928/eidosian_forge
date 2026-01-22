import tensorflow as tf
from tensorboard.summary.v2 import audio  # noqa: F401
from tensorboard.summary.v2 import histogram  # noqa: F401
from tensorboard.summary.v2 import image  # noqa: F401
from tensorboard.summary.v2 import scalar  # noqa: F401
from tensorboard.summary.v2 import text  # noqa: F401
def dynamic_wildcard_import(module):
    """Implements the logic of "from module import *" for the given
        module."""
    symbols = getattr(module, '__all__', None)
    if symbols is None:
        symbols = [k for k in module.__dict__.keys() if not k.startswith('_')]
    globals().update({symbol: getattr(module, symbol) for symbol in symbols})