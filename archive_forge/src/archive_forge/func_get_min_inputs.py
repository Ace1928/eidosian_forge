import numpy as np
from ase import Atoms
def get_min_inputs(self):
    """Returns the number of inputs required for a mutation,
        this is to know how many candidates should be selected
        from the population."""
    return self.min_inputs