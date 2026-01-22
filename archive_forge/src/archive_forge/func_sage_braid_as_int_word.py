from . import links_base, alexander
from .links_base import CrossingStrand, Crossing
from ..sage_helper import _within_sage, sage_method, sage_pd_clockwise
def sage_braid_as_int_word(braid):
    """
    Convert a Sage Braid to a word.
    """
    G = braid.parent()
    gen_idx = {g: i + 1 for i, g in enumerate(G.gens())}
    ans = []
    for g, e in braid.syllables():
        if e > 0:
            ans += e * [gen_idx[g]]
        else:
            ans += abs(e) * [-gen_idx[g]]
    n = G.ngens()
    if max((abs(a) for a in ans)) < n:
        ans += [n, -n]
    return ans