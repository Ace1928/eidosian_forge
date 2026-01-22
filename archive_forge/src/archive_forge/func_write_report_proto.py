import collections
import hashlib
import os
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tensor_tracer_pb2
def write_report_proto(self, report_path, report_proto, tt_parameters):
    """Writes the given report proto under trace_dir."""
    gfile.MakeDirs(tt_parameters.trace_dir)
    with gfile.GFile(report_path, 'wb') as f:
        f.write(report_proto.SerializeToString())