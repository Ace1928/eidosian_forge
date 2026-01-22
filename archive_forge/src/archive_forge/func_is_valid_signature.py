from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import utils_impl as utils
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['saved_model.is_valid_signature', 'saved_model.signature_def_utils.is_valid_signature'])
@deprecation.deprecated_endpoints('saved_model.signature_def_utils.is_valid_signature')
def is_valid_signature(signature_def):
    """Determine whether a SignatureDef can be served by TensorFlow Serving."""
    if signature_def is None:
        return False
    return _is_valid_classification_signature(signature_def) or _is_valid_regression_signature(signature_def) or _is_valid_predict_signature(signature_def)