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
def get_trainable_name(cls, run_object: Union[str, Callable, Type]):
    """Get Trainable name.

        Args:
            run_object: Trainable to run. If string,
                assumes it is an ID and does not modify it. Otherwise,
                returns a string corresponding to the run_object name.

        Returns:
            A string representing the trainable identifier.

        Raises:
            TuneError: if ``run_object`` passed in is invalid.
        """
    from ray.tune.search.sample import Domain
    if isinstance(run_object, str) or isinstance(run_object, Domain):
        return run_object
    elif isinstance(run_object, type) or callable(run_object):
        name = 'DEFAULT'
        if hasattr(run_object, '_name'):
            name = run_object._name
        elif hasattr(run_object, '__name__'):
            fn_name = run_object.__name__
            if fn_name == '<lambda>':
                name = 'lambda'
            elif fn_name.startswith('<'):
                name = 'DEFAULT'
            else:
                name = fn_name
        elif isinstance(run_object, partial) and hasattr(run_object, 'func') and hasattr(run_object.func, '__name__'):
            name = run_object.func.__name__
        else:
            logger.warning('No name detected on trainable. Using {}.'.format(name))
        return name
    else:
        raise TuneError("Improper 'run' - not string nor trainable.")