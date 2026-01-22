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
@DeveloperAPI
def validate_save_restore(trainable_cls: Type, config: Optional[Dict]=None, num_gpus: int=0):
    """Helper method to check if your Trainable class will resume correctly.

    Args:
        trainable_cls: Trainable class for evaluation.
        config: Config to pass to Trainable when testing.
        num_gpus: GPU resources to allocate when testing.
        use_object_store: Whether to save and restore to Ray's object
            store. Recommended to set this to True if planning to use
            algorithms that pause training (i.e., PBT, HyperBand).
    """
    assert ray.is_initialized(), 'Need Ray to be initialized.'
    remote_cls = ray.remote(num_gpus=num_gpus)(trainable_cls)
    trainable_1 = remote_cls.remote(config=config)
    trainable_2 = remote_cls.remote(config=config)
    from ray.air.constants import TRAINING_ITERATION
    for _ in range(3):
        res = ray.get(trainable_1.train.remote())
    assert res.get(TRAINING_ITERATION), 'Validation will not pass because it requires `training_iteration` to be returned.'
    ray.get(trainable_2.restore.remote(trainable_1.save.remote()))
    res = ray.get(trainable_2.train.remote())
    assert res[TRAINING_ITERATION] == 4
    res = ray.get(trainable_2.train.remote())
    assert res[TRAINING_ITERATION] == 5
    return True