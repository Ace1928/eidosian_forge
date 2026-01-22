import copy
import datetime
from functools import partial
import logging
from pathlib import Path
from pickle import PicklingError
import pprint as pp
import traceback
from typing import (
import ray
from ray.exceptions import RpcError
from ray.train import CheckpointConfig, SyncConfig
from ray.train._internal.storage import StorageContext
from ray.tune.error import TuneError
from ray.tune.registry import register_trainable, is_function_trainable
from ray.tune.stopper import CombinedStopper, FunctionStopper, Stopper, TimeoutStopper
from ray.util.annotations import DeveloperAPI, Deprecated
@classmethod
def register_if_needed(cls, run_object: Union[str, Callable, Type]):
    """Registers Trainable or Function at runtime.

        Assumes already registered if run_object is a string.
        Also, does not inspect interface of given run_object.

        Args:
            run_object: Trainable to run. If string,
                assumes it is an ID and does not modify it. Otherwise,
                returns a string corresponding to the run_object name.

        Returns:
            A string representing the trainable identifier.
        """
    from ray.tune.search.sample import Domain
    if isinstance(run_object, str):
        return run_object
    elif isinstance(run_object, Domain):
        logger.warning('Not registering trainable. Resolving as variant.')
        return run_object
    name = cls.get_trainable_name(run_object)
    try:
        register_trainable(name, run_object)
    except (TypeError, PicklingError) as e:
        extra_msg = 'Other options: \n-Try reproducing the issue by calling `pickle.dumps(trainable)`. \n-If the error is typing-related, try removing the type annotations and try again.'
        raise type(e)(str(e) + ' ' + extra_msg) from None
    return name