import io
import os
from tensorflow.lite.toco.logging import toco_conversion_log_pb2 as _toco_conversion_log_pb2
from tensorflow.python.lib.io import file_io as _file_io
from tensorflow.python.platform import resource_loader as _resource_loader
def get_operator_type(op_name, conversion_log):
    if op_name in conversion_log.built_in_ops:
        return 'BUILT-IN'
    elif op_name in conversion_log.custom_ops:
        return 'CUSTOM OP'
    else:
        return 'SELECT OP'