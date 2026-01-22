from absl import logging
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.eager.polymorphic_function import attributes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import function_serialization
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.trackable import base
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
def create_signature_map(signatures):
    """Creates an object containing `signatures`."""
    signature_map = _SignatureMap()
    for name, func in signatures.items():
        assert isinstance(func, defun.ConcreteFunction)
        assert isinstance(func.structured_outputs, collections_abc.Mapping)
        if len(func._arg_keywords) == 1:
            assert 1 == func._num_positional_args
        else:
            assert 0 == func._num_positional_args
        signature_map._add_signature(name, func)
    return signature_map