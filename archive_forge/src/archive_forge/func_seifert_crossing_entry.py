from .ordered_set import OrderedSet
from .simplify import reverse_type_II
from .links_base import Link  # Used for testing only
from .. import ClosedBraid    # Used for testing only
from itertools import combinations
def seifert_crossing_entry(crossing_strand):
    d = (crossing_strand.strand_index, (crossing_strand.strand_index + 2) % 4)
    if d in crossing_strand.crossing.directions:
        return crossing_strand
    entries = crossing_strand.crossing.entry_points()
    entries.remove(crossing_strand.rotate(2))
    return entries[0]