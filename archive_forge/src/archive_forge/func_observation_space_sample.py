from typing import Optional
import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import MultiAgentDict
def observation_space_sample(self, agent_ids: list=None) -> MultiAgentDict:
    sample = self.observation_space.sample()
    if agent_ids is None:
        return sample
    return {aid: sample[aid] for aid in agent_ids}