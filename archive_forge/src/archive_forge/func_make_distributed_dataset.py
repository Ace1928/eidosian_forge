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
def make_distributed_dataset(self, dataset, cluster, processing_mode='parallel_epochs', **kwargs):
    kwargs['task_refresh_interval_hint_ms'] = 20
    if 'data_transfer_protocol' not in kwargs:
        kwargs['data_transfer_protocol'] = self.default_data_transfer_protocol
    if 'compression' not in kwargs:
        kwargs['compression'] = self.default_compression
    return dataset.apply(data_service_ops._distribute(processing_mode, cluster.dispatcher_address(), **kwargs))