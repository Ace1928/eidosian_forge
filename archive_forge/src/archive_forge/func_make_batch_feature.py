import os
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import parsing_ops
from tensorflow.python.util import compat
def make_batch_feature(self, filenames, num_epochs, batch_size, label_key=None, reader_num_threads=1, parser_num_threads=1, shuffle=False, shuffle_seed=None, drop_final_batch=False):
    self.filenames = filenames
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    return readers.make_batched_features_dataset(file_pattern=self.filenames, batch_size=self.batch_size, features={'file': parsing_ops.FixedLenFeature([], dtypes.int64), 'record': parsing_ops.FixedLenFeature([], dtypes.int64), 'keywords': parsing_ops.VarLenFeature(dtypes.string), 'label': parsing_ops.FixedLenFeature([], dtypes.string)}, label_key=label_key, reader=core_readers.TFRecordDataset, num_epochs=self.num_epochs, shuffle=shuffle, shuffle_seed=shuffle_seed, reader_num_threads=reader_num_threads, parser_num_threads=parser_num_threads, drop_final_batch=drop_final_batch)