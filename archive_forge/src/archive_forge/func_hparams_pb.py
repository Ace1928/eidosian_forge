import abc
import hashlib
import json
import random
import time
import numpy as np
from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import metadata
from tensorboard.plugins.hparams import plugin_data_pb2
def hparams_pb(hparams, trial_id=None, start_time_secs=None):
    """Create a summary encoding hyperparameter values for a single trial.

    Args:
      hparams: A `dict` mapping hyperparameters to the values used in this
        trial. Keys should be the names of `HParam` objects used in an
        experiment, or the `HParam` objects themselves. Values should be
        Python `bool`, `int`, `float`, or `string` values, depending on
        the type of the hyperparameter.
      trial_id: An optional `str` ID for the set of hyperparameter values
        used in this trial. Defaults to a hash of the hyperparameters.
      start_time_secs: The time that this trial started training, as
        seconds since epoch. Defaults to the current time.

    Returns:
      A TensorBoard `summary_pb2.Summary` message.
    """
    if start_time_secs is None:
        start_time_secs = time.time()
    hparams = _normalize_hparams(hparams)
    group_name = _derive_session_group_name(trial_id, hparams)
    session_start_info = plugin_data_pb2.SessionStartInfo(group_name=group_name, start_time_secs=start_time_secs)
    for hp_name in sorted(hparams):
        hp_value = hparams[hp_name]
        if isinstance(hp_value, bool):
            session_start_info.hparams[hp_name].bool_value = hp_value
        elif isinstance(hp_value, (float, int)):
            session_start_info.hparams[hp_name].number_value = hp_value
        elif isinstance(hp_value, str):
            session_start_info.hparams[hp_name].string_value = hp_value
        else:
            raise TypeError('hparams[%r] = %r, of unsupported type %r' % (hp_name, hp_value, type(hp_value)))
    return _summary_pb(metadata.SESSION_START_INFO_TAG, plugin_data_pb2.HParamsPluginData(session_start_info=session_start_info))