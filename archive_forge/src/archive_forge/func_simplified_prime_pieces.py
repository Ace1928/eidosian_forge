import random
from .. import graphs
from . import links, twist
from spherogram.planarmap import random_map as raw_random_map
def simplified_prime_pieces(link, simplify_fun):
    cur = link.deconnect_sum(True)
    while cur:
        L = cur.pop()
        if simplify_fun(L):
            cur += L.deconnect_sum(True)
        else:
            yield L