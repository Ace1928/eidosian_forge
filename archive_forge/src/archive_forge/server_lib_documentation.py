import collections
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.data.experimental.service import _pywrap_server_lib
from tensorflow.python.data.experimental.service import _pywrap_utils
from tensorflow.python.util.tf_export import tf_export
Returns the progresses of the snapshot tasks currently being executed.

    Returns:
      An `Iterable[common_pb2.SnapshotTaskProgress]`.
    