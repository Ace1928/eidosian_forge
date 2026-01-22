from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
from concurrent import futures
import threading
import time
import uuid
from absl.testing import parameterized
import numpy as np
from seed_rl.grpc.python import ops
from six.moves import range
import tensorflow as tf
def test_two_clients(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
        return x + 1
    server.bind(foo)
    server.start()
    client = ops.Client(address)
    client2 = ops.Client(address)
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        f0 = executor.submit(client.foo, 42)
        f1 = executor.submit(client2.foo, 44)
        self.assertAllEqual(43, f0.result())
        self.assertAllEqual(45, f1.result())
    server.shutdown()