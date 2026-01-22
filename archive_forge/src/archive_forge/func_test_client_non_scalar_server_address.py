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
def test_client_non_scalar_server_address(self):
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, 'server_address must be a scalar'):
        ops.Client(['localhost:8000', 'localhost:8001'])