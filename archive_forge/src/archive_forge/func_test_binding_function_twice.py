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
def test_binding_function_twice(self):
    address = self.get_unix_address()
    server = ops.Server([address])

    @tf.function(input_signature=[])
    def foo():
        return 42
    server.bind(foo)
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "Function 'foo' was bound twice."):
        server.bind(foo)