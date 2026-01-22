import gymnasium as gym
import numpy as np
from ray.rllib.utils.annotations import PublicAPI
Represents a d - 1 dimensional Simplex in R^d.

    That is, all coordinates are in [0, 1] and sum to 1.
    The dimension d of the simplex is assumed to be shape[-1].

    Additionally one can specify the underlying distribution of
    the simplex as a Dirichlet distribution by providing concentration
    parameters. By default, sampling is uniform, i.e. concentration is
    all 1s.

    Example usage:
    self.action_space = spaces.Simplex(shape=(3, 4))
        --> 3 independent 4d Dirichlet with uniform concentration
    