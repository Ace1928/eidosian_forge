from typing import Generator, Optional, Text
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['tpu.bfloat16_scope'])
@tf_contextlib.contextmanager
def bfloat16_scope(name: Optional[Text]=None) -> Generator[variable_scope.variable_scope, None, None]:
    """Scope class for bfloat16 variables so that the model uses custom getter.

  This enables variables to be read as bfloat16 type when using get_variable.

  Arguments:
    name: Name to use for scope.

  Yields:
    a variable scope.
  """
    if name is None:
        name = ''
    with variable_scope.variable_scope(name, custom_getter=_get_custom_getter()) as varscope:
        yield varscope