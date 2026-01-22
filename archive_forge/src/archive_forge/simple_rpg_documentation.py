import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
from ray.rllib.utils.spaces.repeated import Repeated
Example of a custom env with a complex, structured observation.

    The observation is a list of players, each of which is a Dict of
    attributes, and may further hold a list of items (categorical space).

    Note that the env doesn't train, it's just a dummy example to show how to
    use spaces.Repeated in a custom model (see CustomRPGModel below).
    