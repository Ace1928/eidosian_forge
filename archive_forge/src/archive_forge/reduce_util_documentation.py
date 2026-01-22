import enum
from tensorflow.python.ops import variable_scope
from tensorflow.python.util.tf_export import tf_export
Indicates how a set of values should be reduced.

  * `SUM`: Add all the values.
  * `MEAN`: Take the arithmetic mean ("average") of the values.
  