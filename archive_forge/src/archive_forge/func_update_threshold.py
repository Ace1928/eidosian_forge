from lognormal_around import lognormal_around
import numpy as np
import random
def update_threshold(self):
    """
        Update the neuron's firing threshold based on recent firing history.
        """
    if len(self.firing_history) > 5:
        recent_firings = self.firing_history[-5:]
        if max(recent_firings) - min(recent_firings) < 20:
            self.threshold -= 0.1 * self.global_scaling_factor
        else:
            self.threshold = self.initial_threshold
    self.threshold = max(20, self.threshold)