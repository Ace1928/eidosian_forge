import logging
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import AgentID, EnvID, EpisodeID, PolicyID, TensorType
@abstractmethod
def get_inference_input_dict(self, policy_id: PolicyID) -> Dict[str, TensorType]:
    """Returns an input_dict for an (inference) forward pass from our data.

        The input_dict can then be used for action computations inside a
        Policy via `Policy.compute_actions_from_input_dict()`.

        Args:
            policy_id: The Policy ID to get the input dict for.

        Returns:
            Dict[str, TensorType]: The input_dict to be passed into the ModelV2
                for inference/training.

        .. testcode::
            :skipif: True

            obs, r, terminated, truncated, info = env.step(action)
            collector.add_action_reward_next_obs(12345, 0, "pol0", False, {
                "action": action, "obs": obs, "reward": r,
                "terminated": terminated, "truncated", truncated
            })
            input_dict = collector.get_inference_input_dict(policy.model)
            action = policy.compute_actions_from_input_dict(input_dict)
            # repeat
        """
    raise NotImplementedError