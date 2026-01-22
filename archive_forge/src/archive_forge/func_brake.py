import math
import Box2D
import numpy as np
from gym.error import DependencyNotInstalled
def brake(self, b):
    """control: brake

        Args:
            b (0..1): Degree to which the brakes are applied. More than 0.9 blocks the wheels to zero rotation"""
    for w in self.wheels:
        w.brake = b