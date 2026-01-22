import logging
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import AgentID, EnvID, EpisodeID, PolicyID, TensorType
@abstractmethod
def total_env_steps(self) -> int:
    """Returns total number of env-steps taken so far.

        Thereby, a step in an N-agent multi-agent environment counts as only 1
        for this metric. The returned count contains everything that has not
        been built yet (and returned as MultiAgentBatches by the
        `try_build_truncated_episode_multi_agent_batch` or
        `postprocess_episode(build=True)` methods). After such build, this
        counter is reset to 0.

        Returns:
            int: The number of env-steps taken in total in the environment(s)
                so far.
        """
    raise NotImplementedError