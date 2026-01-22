from lognormal_around import lognormal_around
import numpy as np
import random
def calculate_firing_probability(self, modulated_input):
    """
        Calculate the probability of neuron firing based on the input signal.
        """
    if modulated_input < self.threshold:
        return 0
    else:
        return min(1, (modulated_input - self.threshold) / self.threshold)