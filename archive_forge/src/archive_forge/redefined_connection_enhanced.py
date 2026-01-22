
from lognormal_around import lognormal_around
import random

class RedefinedConnection:
    """
    Redefined Connection class with neurotransmitter effects and variable connection delays.
    """
    def __init__(self, global_scaling_factor=1.0, is_excitatory=True):
        self.strength = lognormal_around(2.75, 0.5, 5) * global_scaling_factor
        self.delay = int(lognormal_around(3, 1, 5))
        self.is_excitatory = is_excitatory  # True for excitatory, False for inhibitory
        self.global_scaling_factor = global_scaling_factor

    def update_strength(self, pre_activity, post_activity):
        """
        Update the connection strength based on neuron activity.
        """
        # Hebbian-like learning rule
        self.strength += (pre_activity * post_activity * 0.01 * self.global_scaling_factor)
        self.strength = max(0, self.strength)  # Ensure strength doesn't go negative

    def get_effective_strength(self):
        """
        Get the effective strength of the connection, considering if it's excitatory or inhibitory.
        """
        if self.is_excitatory:
            return self.strength
        else:
            return -self.strength  # Inhibitory connections have a negative effect
