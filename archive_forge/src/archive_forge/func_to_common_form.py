import logging
from pyomo.core.base import (
from pyomo.mpec.complementarity import Complementarity
from pyomo.gdp import Disjunct
def to_common_form(self, cdata, free_vars):
    """
        Convert a common form that can processed by AMPL
        """
    _e1 = cdata._canonical_expression(cdata._args[0])
    _e2 = cdata._canonical_expression(cdata._args[1])
    if False:
        if _e1[0] is None:
            print(None)
        else:
            print(str(_e1[0]))
        if _e1[1] is None:
            print(None)
        else:
            print(str(_e1[1]))
        if len(_e1) > 2:
            if _e1[2] is None:
                print(None)
            else:
                print(str(_e1[2]))
        if _e2[0] is None:
            print(None)
        else:
            print(str(_e2[0]))
        if _e2[1] is None:
            print(None)
        else:
            print(str(_e2[1]))
        if len(_e2) > 2:
            if _e2[2] is None:
                print(None)
            else:
                print(str(_e2[2]))
    if len(_e1) == 2:
        cdata.c = Constraint(expr=_e1)
        return
    if len(_e2) == 2:
        cdata.c = Constraint(expr=_e2)
        return
    if (_e1[0] is None) + (_e1[2] is None) + (_e2[0] is None) + (_e2[2] is None) != 2:
        raise RuntimeError('Complementarity condition %s must have exactly two finite bounds' % cdata.name)
    if not id(_e2[1]) in free_vars and id(_e1[1]) in free_vars:
        _e1, _e2 = (_e2, _e1)
    if not _e1[0] is None:
        cdata.bv = Var()
        cdata.c = Constraint(expr=0 <= cdata.bv)
        if not _e2[0] is None:
            cdata.bc = Constraint(expr=cdata.bv == _e1[1] - _e1[0])
        else:
            cdata.bc = Constraint(expr=cdata.bv == _e1[0] - _e1[1])
    elif not _e1[2] is None:
        cdata.bv = Var()
        cdata.c = Constraint(expr=0 <= cdata.bv)
        if not _e2[2] is None:
            cdata.bc = Constraint(expr=cdata.bv == _e1[1] - _e1[2])
        else:
            cdata.bc = Constraint(expr=cdata.bv == _e1[2] - _e1[1])
    else:
        cdata.bv = Var()
        cdata.bc = Constraint(expr=cdata.bv == _e1[1])
        cdata.c = Constraint(expr=(None, cdata.bv, None))
    if id(_e2[1]) in free_vars:
        var = _e2[1]
        cdata.c._vid = id(_e2[1])
        del free_vars[cdata.c._vid]
    else:
        var = cdata.v = Var()
        cdata.c._vid = id(cdata.v)
        cdata.e = Constraint(expr=cdata.v == _e2[1])
    cdata.c._complementarity = 0
    if not _e2[0] is None:
        if var.lb is None or value(_e2[0]) > value(var.lb):
            var.setlb(_e2[0])
        cdata.c._complementarity += 1
    if not _e2[2] is None:
        if var.ub is None or value(_e2[2]) > value(var.ub):
            var.setub(_e2[2])
        cdata.c._complementarity += 2