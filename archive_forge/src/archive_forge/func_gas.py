import math
import Box2D
import numpy as np
from gym.error import DependencyNotInstalled
def gas(self, gas):
    """control: rear wheel drive

        Args:
            gas (float): How much gas gets applied. Gets clipped between 0 and 1.
        """
    gas = np.clip(gas, 0, 1)
    for w in self.wheels[2:4]:
        diff = gas - w.gas
        if diff > 0.1:
            diff = 0.1
        w.gas += diff