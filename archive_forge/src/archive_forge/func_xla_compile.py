import contextlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export
def xla_compile(node_def):
    return attr_value_pb2.AttrValue(b=compile_ops(node_def))