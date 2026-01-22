import uuid
import gymnasium as gym
from typing import Optional
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.env.external_env import ExternalEnv, _ExternalEnvEpisode
from ray.rllib.utils.typing import MultiAgentDict
@PublicAPI
@override(ExternalEnv)
def start_episode(self, episode_id: Optional[str]=None, training_enabled: bool=True) -> str:
    if episode_id is None:
        episode_id = uuid.uuid4().hex
    if episode_id in self._finished:
        raise ValueError('Episode {} has already completed.'.format(episode_id))
    if episode_id in self._episodes:
        raise ValueError('Episode {} is already started'.format(episode_id))
    self._episodes[episode_id] = _ExternalEnvEpisode(episode_id, self._results_avail_condition, training_enabled, multiagent=True)
    return episode_id