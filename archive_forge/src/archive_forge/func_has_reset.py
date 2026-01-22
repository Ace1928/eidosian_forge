import gym
from gym.error import ResetNeeded
@property
def has_reset(self):
    """Returns if the environment has been reset before."""
    return self._has_reset