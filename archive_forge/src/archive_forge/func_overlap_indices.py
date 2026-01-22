from .links_base import Strand, Crossing, Link
import random
import collections
def overlap_indices(self, crossing):
    neighbors = [cs.opposite() for cs in crossing.crossing_strands()]
    return [self[ns] for ns in neighbors if ns in self]