import gast
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.autograph.pyct import templates
Converts slicing operations to their TF counterpart.

  Currently, relying on the default slice operator that Tensor uses is
  insufficient, because TensorArray and tensor lists use dedicated index read
  and write functions.
  