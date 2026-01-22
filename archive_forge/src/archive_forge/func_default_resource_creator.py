import contextlib
import copy
import weakref
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.trackable import base
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
def default_resource_creator(next_creator, *a, **kw):
    assert next_creator is None
    obj = cls.__new__(cls, *a, **kw)
    obj.__init__(*a, **kw)
    return obj