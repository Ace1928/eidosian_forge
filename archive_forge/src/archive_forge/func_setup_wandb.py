import enum
import os
import pickle
import urllib
import warnings
import numpy as np
from numbers import Number
import pyarrow.fs
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import ray
from ray import logger
from ray.air import session
from ray.air._internal import usage as air_usage
from ray.air.util.node import _force_on_current_node
from ray.tune.logger import LoggerCallback
from ray.tune.utils import flatten_dict
from ray.tune.experiment import Trial
from ray.train._internal.syncer import DEFAULT_SYNC_TIMEOUT
from ray._private.storage import _load_class
from ray.util import PublicAPI
from ray.util.queue import Queue
@PublicAPI(stability='alpha')
def setup_wandb(config: Optional[Dict]=None, api_key: Optional[str]=None, api_key_file: Optional[str]=None, rank_zero_only: bool=True, **kwargs) -> Union[Run, RunDisabled]:
    """Set up a Weights & Biases session.

    This function can be used to initialize a Weights & Biases session in a
    (distributed) training or tuning run.

    By default, the run ID is the trial ID, the run name is the trial name, and
    the run group is the experiment name. These settings can be overwritten by
    passing the respective arguments as ``kwargs``, which will be passed to
    ``wandb.init()``.

    In distributed training with Ray Train, only the zero-rank worker will initialize
    wandb. All other workers will return a disabled run object, so that logging is not
    duplicated in a distributed run. This can be disabled by passing
    ``rank_zero_only=False``, which will then initialize wandb in every training
    worker.

    The ``config`` argument will be passed to Weights and Biases and will be logged
    as the run configuration.

    If no API key or key file are passed, wandb will try to authenticate
    using locally stored credentials, created for instance by running ``wandb login``.

    Keyword arguments passed to ``setup_wandb()`` will be passed to
    ``wandb.init()`` and take precedence over any potential default settings.

    Args:
        config: Configuration dict to be logged to Weights and Biases. Can contain
            arguments for ``wandb.init()`` as well as authentication information.
        api_key: API key to use for authentication with Weights and Biases.
        api_key_file: File pointing to API key for with Weights and Biases.
        rank_zero_only: If True, will return an initialized session only for the
            rank 0 worker in distributed training. If False, will initialize a
            session for all workers.
        kwargs: Passed to ``wandb.init()``.

    Example:

        .. code-block: python

            from ray.air.integrations.wandb import setup_wandb

            def training_loop(config):
                wandb = setup_wandb(config)
                # ...
                wandb.log({"loss": 0.123})

    """
    if not wandb:
        raise RuntimeError('Wandb was not found - please install with `pip install wandb`')
    try:
        _session = session._get_session(warn=False)
        if _session and rank_zero_only and (session.get_world_rank() != 0):
            return RunDisabled()
        default_trial_id = session.get_trial_id()
        default_trial_name = session.get_trial_name()
        default_experiment_name = session.get_experiment_name()
    except RuntimeError:
        default_trial_id = None
        default_trial_name = None
        default_experiment_name = None
    wandb_init_kwargs = {'trial_id': kwargs.get('trial_id') or default_trial_id, 'trial_name': kwargs.get('trial_name') or default_trial_name, 'group': kwargs.get('group') or default_experiment_name}
    wandb_init_kwargs.update(kwargs)
    return _setup_wandb(config=config, api_key=api_key, api_key_file=api_key_file, **wandb_init_kwargs)