from copy import deepcopy
def remove_population(self, pos):
    """Remove a population (by position)."""
    del self.populations[pos]