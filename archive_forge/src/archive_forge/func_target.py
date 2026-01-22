import collections
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.data.experimental.service import _pywrap_server_lib
from tensorflow.python.data.experimental.service import _pywrap_utils
from tensorflow.python.util.tf_export import tf_export
@property
def target(self):
    """Returns a target that can be used to connect to the server.

    >>> dispatcher = tf.data.experimental.service.DispatchServer()
    >>> dataset = tf.data.Dataset.range(10)
    >>> dataset = dataset.apply(tf.data.experimental.service.distribute(
    ...     processing_mode="parallel_epochs", service=dispatcher.target))

    The returned string will be in the form protocol://address, e.g.
    "grpc://localhost:5050".
    """
    return '{0}://localhost:{1}'.format(self._config.protocol, self._server.bound_port())