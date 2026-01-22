import warnings
import numpy as np
from .tools import (
from .initialization import Initialization
from . import tools
def initialize_approximate_diffuse(self, variance=None):
    """
        Initialize the statespace model with approximate diffuse values.

        Rather than following the exact diffuse treatment (which is developed
        for the case that the variance becomes infinitely large), this assigns
        an arbitrary large number for the variance.

        Parameters
        ----------
        variance : float, optional
            The variance for approximating diffuse initial conditions. Default
            is 1e6.
        """
    if variance is None:
        variance = self.initial_variance
    self.initialize('approximate_diffuse', approximate_diffuse_variance=variance)