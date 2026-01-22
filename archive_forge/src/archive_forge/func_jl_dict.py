from collections import OrderedDict
from chempy import Reaction
from chempy.kinetics.rates import MassAction, RadiolyticBase
from chempy.units import to_unitless, default_units as u
def jl_dict(od):
    return 'Dict([%s])' % ', '.join(['(:%s, %.4g)' % (k, v) for k, v in od.items()])