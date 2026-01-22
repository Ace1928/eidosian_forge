from gymnasium.spaces import Space
from typing import Optional
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.exploration.gaussian_noise import GaussianNoise
from ray.rllib.utils.schedules import ConstantSchedule

        Args:
            action_space: The gym action space used by the environment.
            num_workers: The overall number of workers used.
            worker_index: The index of the Worker using this
                Exploration.
            framework: One of None, "tf", "torch".
        