import random
import timeit
from functools import wraps
from typing import Callable, Optional
from ..configuration_utils import PretrainedConfig
from ..models.auto.modeling_tf_auto import TF_MODEL_MAPPING, TF_MODEL_WITH_LM_HEAD_MAPPING
from ..utils import is_py3nvml_available, is_tf_available, logging
from .benchmark_utils import (
def random_input_ids(batch_size: int, sequence_length: int, vocab_size: int) -> ['tf.Tensor']:
    rng = random.Random()
    values = [rng.randint(0, vocab_size - 1) for i in range(batch_size * sequence_length)]
    return tf.constant(values, shape=(batch_size, sequence_length), dtype=tf.int32)