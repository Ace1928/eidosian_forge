from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from six.moves import range
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.checkpoint import saveable_compat
def size(self, name=None):
    with ops.name_scope(name, 'sharded_mutable_hash_table_size'):
        sizes = [self._table_shards[i].size() for i in range(self._num_shards)]
        return tf.math.add_n(sizes)