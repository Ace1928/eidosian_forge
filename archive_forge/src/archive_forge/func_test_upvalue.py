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
def test_upvalue(self):
    address = self.get_unix_address()
    server = ops.Server([address])
    a = tf.constant(2)

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def foo(x):
        return x / a
    server.bind(foo)
    server.start()
    client = ops.Client(address)
    self.assertAllEqual(21, client.foo(42))
    server.shutdown()