import random
from .. import graphs
from . import links, twist
from spherogram.planarmap import random_map as raw_random_map
def largest_prime_piece(link, simplify_fun):
    pieces = simplified_prime_pieces(link, simplify_fun)
    return max(pieces, key=lambda L: len(L.crossings), default=links.Link([]))