import random
import timeit
from functools import wraps
from typing import Callable, Optional
from ..configuration_utils import PretrainedConfig
from ..models.auto.modeling_tf_auto import TF_MODEL_MAPPING, TF_MODEL_WITH_LM_HEAD_MAPPING
from ..utils import is_py3nvml_available, is_tf_available, logging
from .benchmark_utils import (
def run_func(func):

    @wraps(func)
    def run_in_eager_mode(*args, **kwargs):
        return func(*args, **kwargs)

    @wraps(func)
    @tf.function(experimental_compile=use_xla)
    def run_in_graph_mode(*args, **kwargs):
        return func(*args, **kwargs)
    if do_eager_mode is True:
        if use_xla is not False:
            raise ValueError('Cannot run model in XLA, if `args.eager_mode` is set to `True`. Please set `args.eager_mode=False`.')
        return run_in_eager_mode
    else:
        return run_in_graph_mode