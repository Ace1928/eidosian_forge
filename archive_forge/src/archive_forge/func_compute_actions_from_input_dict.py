import gymnasium as gym
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import AlgorithmConfigDict
@override(RandomPolicy)
def compute_actions_from_input_dict(self, *args, **kwargs):
    if self._leakage_size == 'small':
        self._leak.append(1)
    else:
        self._leak.append([1] * 100)
    return super().compute_actions_from_input_dict(*args, **kwargs)