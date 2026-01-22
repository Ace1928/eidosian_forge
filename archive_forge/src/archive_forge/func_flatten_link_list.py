from ..snap.t3mlite import simplex
from ..hyperboloid import *
def flatten_link_list(x):
    y = x
    l = []
    while True:
        l.append(y)
        y = y.next_
        if x is y:
            return l