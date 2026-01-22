import tempfile
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
def restart(self, use_same_port=True):
    """Restarts the worker, stopping it first if it is already running."""
    if self._running:
        self.stop()
    port = 0
    if use_same_port:
        port = self._port
    self._server = _make_worker(self._dispatcher_address, self._protocol, self._data_transfer_protocol, self._shutdown_quiet_period_ms, port)
    self._server.start()
    self._port = int(self._server._address.split(':')[1])
    self._running = True