import collections
import itertools
from heat.common import exception
def requirer_edges(rqr):
    return itertools.chain([(rqr, key)], get_edges(rqr))