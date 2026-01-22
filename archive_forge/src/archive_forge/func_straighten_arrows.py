from .ordered_set import OrderedSet
from .simplify import reverse_type_II
from .links_base import Link  # Used for testing only
from .. import ClosedBraid    # Used for testing only
from itertools import combinations
def straighten_arrows(arrows):
    totally_straightened = False
    while not totally_straightened:
        totally_straightened = True
        for arrow in arrows:
            tail, head = (arrow[0], arrow[1])
            if tail < head:
                diff = head - tail
                same_start_strand = [x for x in arrows if x[2] == arrow[2] and x[0] >= tail]
                for other_arrow in same_start_strand:
                    other_arrow[0] += diff
                one_strand_behind = [x for x in arrows if x[2] == arrow[2] - 1 and x[1] >= tail]
                for other_arrow in one_strand_behind:
                    other_arrow[1] += diff
                totally_straightened = False
            elif head < tail:
                diff = tail - head
                same_end_strand = [x for x in arrows if x[2] == arrow[2] and x[1] >= head]
                for other_arrow in same_end_strand:
                    other_arrow[1] += diff
                one_strand_ahead = [x for x in arrows if x[2] == arrow[2] + 1 and x[0] >= head]
                for other_arrow in one_strand_ahead:
                    other_arrow[0] += diff
                totally_straightened = False