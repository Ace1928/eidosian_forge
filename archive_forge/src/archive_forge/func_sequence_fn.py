import wrapt as _wrapt
from tensorflow.python.util import _pywrap_nest
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest_util
from tensorflow.python.util.compat import collections_abc as _collections_abc
from tensorflow.python.util.tf_export import tf_export
def sequence_fn(instance, args):
    if isinstance(instance, list):
        return tuple(args)
    return nest_util.sequence_like(instance, args)