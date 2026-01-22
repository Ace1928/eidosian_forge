from .links_base import Strand, Crossing, Link
import random
import collections
def overlap_is_consecutive(self, crossing):
    overlap = self.overlap_indices(crossing)
    return len(overlap) > 0 and is_range(overlap)