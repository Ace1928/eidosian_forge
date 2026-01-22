import copy
import functools
import re
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import initializers
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
def print_dtensor_variable_summary(model, print_fn, line_length):
    if getattr(model, '_layout_map', None) is not None:
        mesh = model._layout_map.get_default_mesh()
    elif hasattr(model, 'distribute_strategy') and hasattr(model.distribute_strategy, '_mesh'):
        mesh = model.distribute_strategy._mesh
    else:
        mesh = None
    if mesh:
        total_weight_count, total_memory_size, per_sharing_spec_result = dtensor_variable_summary(model.weights)
        total_per_device_memory_size = 0
        for sharding_spec in sorted(per_sharing_spec_result.keys()):
            count, memory_size = per_sharing_spec_result[sharding_spec]
            if len(sharding_spec) == 0:
                print_fn(f'{count} / {total_weight_count} params ({readable_memory_size(memory_size)}) are fully replicated')
                per_device_size = memory_size
            else:
                sharding_factor = np.prod([mesh.dim_size(s) for s in sharding_spec])
                per_device_size = memory_size / sharding_factor
                print_fn(f"{count} / {total_weight_count} params ({readable_memory_size(memory_size)}) are sharded based on spec '{sharding_spec}' and across {sharding_factor} devices.")
            total_per_device_memory_size += per_device_size
        print_fn(f'Overall per device memory usage: {readable_memory_size(total_per_device_memory_size)}')
        print_fn('Overall sharding factor: {:.2f}'.format(total_memory_size / total_per_device_memory_size))
        print_fn('_' * line_length)