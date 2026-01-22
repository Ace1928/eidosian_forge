from abc import ABC
import numpy as np
from ray.rllib.examples.env.utils.interfaces import InfoAccumulationInterface

    Mixin class to add logging capability in N player games with
    discrete actions.
    Logs the frequency of action profiles used
    (action profile: the set of actions used during one step by all players).
    