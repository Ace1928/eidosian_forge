import argparse
import sys
from absl import app
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
Parse content of cluster_spec string and inject info into cluster protobuf.

  Args:
    cluster_spec: cluster specification string, e.g.,
          "local|localhost:2222;localhost:2223"
    cluster: cluster protobuf.
    verbose: If verbose logging is requested.

  Raises:
    ValueError: if the cluster_spec string is invalid.
  