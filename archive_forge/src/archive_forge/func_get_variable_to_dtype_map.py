from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.util import compat
from tensorflow.python.util._pywrap_checkpoint_reader import CheckpointReader
from tensorflow.python.util.tf_export import tf_export
def get_variable_to_dtype_map(self):
    return {name: dtypes.DType(type_enum) for name, type_enum in self._GetVariableToDataTypeMap().items()}