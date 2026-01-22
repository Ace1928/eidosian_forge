from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from contextlib import contextmanager
import copy
import tensorflow as tf
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.tpu import device_assignment as tpu_device_assignment
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import tpu_config
@property
def embedding_config(self):
    """Returns the embedding config based on current mode."""
    master = self._get_master_address()
    if master in self._lazy_embedding_config_dict:
        embedding_config = self._lazy_embedding_config_dict[master]
    else:
        embedding_config = None
        if self._use_tpu and self._embedding_config_spec:
            embedding_config = _tpu_estimator_embedding.EmbeddingConfig(self._embedding_config_spec, self._train_batch_size, self._eval_batch_size, self.num_hosts, self.num_cores, self.config)
            if not embedding_config.has_embedding_tables():
                embedding_config = None
        self._lazy_embedding_config_dict[master] = embedding_config
    if embedding_config is not None:
        mode = self._assert_mode()
        embedding_config.tpu_embedding = embedding_config.get_tpu_embedding(mode)
    return embedding_config