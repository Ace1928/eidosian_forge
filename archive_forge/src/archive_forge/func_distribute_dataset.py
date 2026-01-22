import collections
import contextlib
import os
import re
import warnings
import numpy as np
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import distribution_lib
from keras.src.backend.common import global_state
def distribute_dataset(self, dataset):
    from tensorflow.python.data.experimental.ops import distribute as tf_data_distribute
    from keras.src.utils.module_utils import tensorflow as tf
    if not isinstance(dataset, tf.data.Dataset):
        raise ValueError(f'Only `tf.data.Dataset` is supported for sharding, got {type(dataset)}')
    if not self._is_multi_process:
        return dataset
    global_batch_size = tf_data_distribute.compute_batch_size(dataset)
    if global_batch_size.numpy() < 0:
        raise ValueError('The batch size of the input dataset is unknown. Please config the batch size for the input dataset, e.g via `dataset.batch(batch_size)`')
    mesh_batch_dim_index = self.device_mesh.axis_names.index(self._batch_dim_name)
    mesh_batch_dim_size = self.device_mesh.shape[mesh_batch_dim_index]
    local_device_count = np.prod(self.device_mesh.shape) // self._num_process
    if mesh_batch_dim_size < local_device_count:
        return dataset.prefetch(tf.data.AUTOTUNE)
    else:
        if mesh_batch_dim_size % local_device_count != 0:
            raise ValueError(f'The Batch dimention of the mesh is not compatible with the local worker device count. Mesh batch dim = {mesh_batch_dim_size} and local device count = {local_device_count}')
        num_shards = mesh_batch_dim_size // local_device_count
        per_worker_batch_size = global_batch_size // num_shards
        distributed_dataset = dataset.rebatch(per_worker_batch_size)
        distributed_dataset = tf_data_distribute._AutoShardDataset(distributed_dataset, num_workers=num_shards, index=self._process_id % num_shards, num_replicas=num_shards)
        return distributed_dataset.prefetch(tf.data.AUTOTUNE)