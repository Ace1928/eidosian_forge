import networkx as nx
from .. import t3mlite as t3m
from ..t3mlite.simplex import *
from . import surface
def doctest_globals():
    import snappy.snap.t3mlite
    return {'Mcomplex': snappy.snap.t3mlite.Mcomplex}