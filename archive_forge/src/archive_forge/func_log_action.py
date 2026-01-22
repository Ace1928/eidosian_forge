import uuid
import gymnasium as gym
from typing import Optional
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.env.external_env import ExternalEnv, _ExternalEnvEpisode
from ray.rllib.utils.typing import MultiAgentDict
@PublicAPI
@override(ExternalEnv)
def log_action(self, episode_id: str, observation_dict: MultiAgentDict, action_dict: MultiAgentDict) -> None:
    """Record an observation and (off-policy) action taken.

        Args:
            episode_id: Episode id returned from start_episode().
            observation_dict: Current environment observation.
            action_dict: Action for the observation.
        """
    episode = self._get(episode_id)
    episode.log_action(observation_dict, action_dict)