import abc
import time
from typing import (
import numpy as np
from ray.data._internal.block_batching.iter_batches import iter_batches
from ray.data._internal.stats import DatasetStats, StatsManager
from ray.data.block import (
from ray.types import ObjectRef
from ray.util.annotations import PublicAPI
def convert_batch_to_tensors(batch: Dict[str, np.ndarray], *, columns: Union[str, List[str]], type_spec: Union[tf.TypeSpec, Dict[str, tf.TypeSpec]]) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
    if isinstance(columns, str):
        return convert_ndarray_to_tf_tensor(batch[columns], type_spec=type_spec)
    return {column: convert_ndarray_to_tf_tensor(batch[column], type_spec=type_spec[column]) for column in columns}