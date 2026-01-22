from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.summary.writer import writer
from tensorflow.python.summary.writer import writer_cache
@classmethod
def uninstall(cls):
    if not cls._replaced_summary_writer:
        raise ValueError('FakeSummaryWriter not installed.')
    writer.FileWriter = cls._replaced_summary_writer
    writer_cache.FileWriter = cls._replaced_summary_writer
    cls._replaced_summary_writer = None