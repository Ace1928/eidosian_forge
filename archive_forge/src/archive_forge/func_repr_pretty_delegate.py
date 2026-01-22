import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def repr_pretty_delegate(obj):
    if optional_dep_ok and 'IPython' in sys.modules:
        from IPython.lib.pretty import pretty
        return pretty(obj)
    else:
        return _mini_pretty(obj)