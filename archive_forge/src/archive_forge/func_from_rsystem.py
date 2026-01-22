from collections import OrderedDict
from chempy import Reaction
from chempy.kinetics.rates import MassAction, RadiolyticBase
from chempy.units import to_unitless, default_units as u
@classmethod
def from_rsystem(cls, rsys, par_vals, *, variables=None, substance_key_map=lambda i, sk: 'y%d' % i, **kwargs):
    if not isinstance(substance_key_map, dict):
        substance_key_map = {sk: substance_key_map(si, sk) for si, sk in enumerate(rsys.substances)}
    parmap = dict([(r.param.unique_keys[0], 'p%d' % i) for i, r in enumerate(rsys.rxns)])
    rxs, pars = ([], OrderedDict())
    for r in rsys.rxns:
        rs, pk, pv = _r(r, par_vals, substance_key_map, parmap, variables=variables, unit_conc=kwargs.get('unit_conc', cls.defaults['unit_conc']), unit_time=kwargs.get('unit_time', cls.defaults['unit_time']))
        rxs.append(rs)
        if pk in pars:
            raise ValueError('Are you sure (sometimes intentional)?')
        pars[parmap[pk]] = pv
    return cls(rxs=rxs, pars=pars, substance_key_map=substance_key_map, parmap=parmap, **kwargs)