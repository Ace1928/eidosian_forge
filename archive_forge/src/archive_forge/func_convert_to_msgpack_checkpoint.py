import logging
import json
import os
from packaging import version
import re
from typing import Any, Dict, Union
import ray
from ray.rllib.utils.serialization import NOT_SERIALIZABLE, serialize_type
from ray.train import Checkpoint
from ray.util import log_once
from ray.util.annotations import PublicAPI
@PublicAPI(stability='beta')
def convert_to_msgpack_checkpoint(checkpoint: Union[str, Checkpoint], msgpack_checkpoint_dir: str) -> str:
    """Converts an Algorithm checkpoint (pickle based) to a msgpack based one.

    Msgpack has the advantage of being python version independent.

    Args:
        checkpoint: The directory, in which to find the Algorithm checkpoint (pickle
            based).
        msgpack_checkpoint_dir: The directory, in which to create the new msgpack
            based checkpoint.

    Returns:
        The directory in which the msgpack checkpoint has been created. Note that
        this is the same as `msgpack_checkpoint_dir`.
    """
    from ray.rllib.algorithms import Algorithm
    from ray.rllib.utils.policy import validate_policy_id
    msgpack = try_import_msgpack(error=True)
    algo = Algorithm.from_checkpoint(checkpoint)
    state = algo.__getstate__()
    state['algorithm_class'] = serialize_type(state['algorithm_class'])
    state['config'] = state['config'].serialize()
    policy_states = {}
    if 'worker' in state and 'policy_states' in state['worker']:
        policy_states = state['worker'].pop('policy_states', {})
    state['worker']['policy_mapping_fn'] = NOT_SERIALIZABLE
    state['worker']['is_policy_to_train'] = NOT_SERIALIZABLE
    if state['config']['_enable_new_api_stack']:
        state['checkpoint_version'] = str(CHECKPOINT_VERSION_LEARNER)
    else:
        state['checkpoint_version'] = str(CHECKPOINT_VERSION)
    state_file = os.path.join(msgpack_checkpoint_dir, 'algorithm_state.msgpck')
    with open(state_file, 'wb') as f:
        msgpack.dump(state, f)
    with open(os.path.join(msgpack_checkpoint_dir, 'rllib_checkpoint.json'), 'w') as f:
        json.dump({'type': 'Algorithm', 'checkpoint_version': state['checkpoint_version'], 'format': 'msgpack', 'state_file': state_file, 'policy_ids': list(policy_states.keys()), 'ray_version': ray.__version__, 'ray_commit': ray.__commit__}, f)
    for pid, policy_state in policy_states.items():
        validate_policy_id(pid, error=True)
        policy_dir = os.path.join(msgpack_checkpoint_dir, 'policies', pid)
        os.makedirs(policy_dir, exist_ok=True)
        policy = algo.get_policy(pid)
        policy.export_checkpoint(policy_dir, policy_state=policy_state, checkpoint_format='msgpack')
    algo.stop()
    return msgpack_checkpoint_dir