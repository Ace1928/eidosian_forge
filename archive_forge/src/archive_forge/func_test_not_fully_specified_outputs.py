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
def test_not_fully_specified_outputs(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)])
    def foo(x):
        if tf.equal(x[0], 0):
            return tf.zeros([])
        elif tf.equal(x[0], 1):
            return tf.zeros([2])
        else:
            return tf.zeros([1])
    server.bind(foo)
    server.start()

    def client():
        client = ops.Client(address)
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, 'Output must be at least rank 1 when batching is enabled'):
            client.foo(0)
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(client)
        f2 = executor.submit(client)
        f1.result()
        f2.result()
    server.shutdown()