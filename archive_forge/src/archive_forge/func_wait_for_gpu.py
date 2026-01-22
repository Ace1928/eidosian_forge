import copy
import glob
import inspect
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from numbers import Number
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import numpy as np
import psutil
import ray
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.air._internal.json import SafeFallbackEncoder  # noqa
from ray.air._internal.util import (  # noqa: F401
from ray._private.dict import (  # noqa: F401
@PublicAPI(stability='beta')
def wait_for_gpu(gpu_id: Optional[Union[int, str]]=None, target_util: float=0.01, retry: int=20, delay_s: int=5, gpu_memory_limit: Optional[float]=None):
    """Checks if a given GPU has freed memory.

    Requires ``gputil`` to be installed: ``pip install gputil``.

    Args:
        gpu_id: GPU id or uuid to check.
            Must be found within GPUtil.getGPUs(). If none, resorts to
            the first item returned from `ray.get_gpu_ids()`.
        target_util: The utilization threshold to reach to unblock.
            Set this to 0 to block until the GPU is completely free.
        retry: Number of times to check GPU limit. Sleeps `delay_s`
            seconds between checks.
        delay_s: Seconds to wait before check.

    Returns:
        bool: True if free.

    Raises:
        RuntimeError: If GPUtil is not found, if no GPUs are detected
            or if the check fails.

    Example:

    .. code-block:: python

        def tune_func(config):
            tune.util.wait_for_gpu()
            train()

        tuner = tune.Tuner(
            tune.with_resources(
                tune_func,
                resources={"gpu": 1}
            ),
            tune_config=tune.TuneConfig(num_samples=10)
        )
        tuner.fit()

    """
    GPUtil = _import_gputil()
    if GPUtil is None:
        raise RuntimeError('GPUtil must be installed if calling `wait_for_gpu`.')
    if gpu_id is None:
        gpu_id_list = ray.get_gpu_ids()
        if not gpu_id_list:
            raise RuntimeError('No GPU ids found from `ray.get_gpu_ids()`. Did you set Tune resources correctly?')
        gpu_id = gpu_id_list[0]
    gpu_attr = 'id'
    if isinstance(gpu_id, str):
        if gpu_id.isdigit():
            gpu_id = int(gpu_id)
        else:
            gpu_attr = 'uuid'
    elif not isinstance(gpu_id, int):
        raise ValueError(f'gpu_id ({type(gpu_id)}) must be type str/int.')

    def gpu_id_fn(g):
        return getattr(g, gpu_attr)
    gpu_ids = {gpu_id_fn(g) for g in GPUtil.getGPUs()}
    if gpu_id not in gpu_ids:
        raise ValueError(f"{gpu_id} not found in set of available GPUs: {gpu_ids}. `wait_for_gpu` takes either GPU ordinal ID (e.g., '0') or UUID (e.g., 'GPU-04546190-b68d-65ac-101b-035f8faed77d').")
    for i in range(int(retry)):
        gpu_object = next((g for g in GPUtil.getGPUs() if gpu_id_fn(g) == gpu_id))
        if gpu_object.memoryUtil > target_util:
            logger.info(f'Waiting for GPU util to reach {target_util}. Util: {gpu_object.memoryUtil:0.3f}')
            time.sleep(delay_s)
        else:
            return True
    raise RuntimeError('GPU memory was not freed.')