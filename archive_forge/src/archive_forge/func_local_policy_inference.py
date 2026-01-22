import gymnasium as gym
import logging
import numpy as np
import re
from typing import (
import tree  # pip install dm_tree
import ray.cloudpickle as pickle
from ray.rllib.models.preprocessors import ATARI_OBS_SHAPE
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import (
from ray.util import log_once
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
def local_policy_inference(policy: 'Policy', env_id: str, agent_id: str, obs: TensorStructType, reward: Optional[float]=None, terminated: Optional[bool]=None, truncated: Optional[bool]=None, info: Optional[Mapping]=None, explore: bool=None, timestep: Optional[int]=None) -> TensorStructType:
    """Run a connector enabled policy using environment observation.

    policy_inference manages policy and agent/action connectors,
    so the user does not have to care about RNN state buffering or
    extra fetch dictionaries.
    Note that connectors are intentionally run separately from
    compute_actions_from_input_dict(), so we can have the option
    of running per-user connectors on the client side in a
    server-client deployment.

    Args:
        policy: Policy object used in inference.
        env_id: Environment ID. RLlib builds environments' trajectories internally with
            connectors based on this, i.e. one trajectory per (env_id, agent_id) tuple.
        agent_id: Agent ID. RLlib builds agents' trajectories internally with connectors
            based on this, i.e. one trajectory per (env_id, agent_id) tuple.
        obs: Environment observation to base the action on.
        reward: Reward that is potentially used during inference. If not required,
            may be left empty. Some policies have ViewRequirements that require this.
            This can be set to zero at the first inference step - for example after
            calling gmy.Env.reset.
        terminated: `Terminated` flag that is potentially used during inference. If not
            required, may be left None. Some policies have ViewRequirements that
            require this extra information.
        truncated: `Truncated` flag that is potentially used during inference. If not
            required, may be left None. Some policies have ViewRequirements that
            require this extra information.
        info: Info that is potentially used durin inference. If not required,
            may be left empty. Some policies have ViewRequirements that require this.
        explore: Whether to pick an exploitation or exploration action
            (default: None -> use self.config["explore"]).
        timestep: The current (sampling) time step.

    Returns:
        List of outputs from policy forward pass.
    """
    assert policy.agent_connectors, 'policy_inference only works with connector enabled policies.'
    __check_atari_obs_space(obs)
    policy.agent_connectors.in_eval()
    policy.action_connectors.in_eval()
    input_dict = {SampleBatch.NEXT_OBS: obs}
    if reward is not None:
        input_dict[SampleBatch.REWARDS] = reward
    if terminated is not None:
        input_dict[SampleBatch.TERMINATEDS] = terminated
    if truncated is not None:
        input_dict[SampleBatch.TRUNCATEDS] = truncated
    if info is not None:
        input_dict[SampleBatch.INFOS] = info
    acd_list: List[AgentConnectorDataType] = [AgentConnectorDataType(env_id, agent_id, input_dict)]
    ac_outputs: List[AgentConnectorsOutput] = policy.agent_connectors(acd_list)
    outputs = []
    for ac in ac_outputs:
        policy_output = policy.compute_actions_from_input_dict(ac.data.sample_batch, explore=explore, timestep=timestep)
        policy_output = tree.map_structure(lambda x: x[0], policy_output)
        action_connector_data = ActionConnectorDataType(env_id, agent_id, ac.data.raw_dict, policy_output)
        if policy.action_connectors:
            acd = policy.action_connectors(action_connector_data)
            actions = acd.output
        else:
            actions = policy_output[0]
        outputs.append(actions)
        policy.agent_connectors.on_policy_output(action_connector_data)
    return outputs