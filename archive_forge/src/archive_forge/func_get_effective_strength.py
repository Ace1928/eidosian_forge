from lognormal_around import lognormal_around
import random
def get_effective_strength(self):
    """
        Get the effective strength of the connection, considering if it's excitatory or inhibitory.
        """
    if self.is_excitatory:
        return self.strength
    else:
        return -self.strength