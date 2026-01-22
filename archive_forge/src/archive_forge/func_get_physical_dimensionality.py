from functools import reduce
from operator import mul
import sys
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util.pyutil import NameSpace, deprecated
def get_physical_dimensionality(value):
    if is_unitless(value):
        return {}
    _quantities_mapping = {pq.UnitLength: 'length', pq.UnitMass: 'mass', pq.UnitTime: 'time', pq.UnitCurrent: 'current', pq.UnitTemperature: 'temperature', pq.UnitLuminousIntensity: 'luminous_intensity', pq.UnitSubstance: 'amount'}
    return {_quantities_mapping[k.__class__]: v for k, v in uniform(value).simplified.dimensionality.items()}