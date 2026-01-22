import logging
from io import StringIO
from operator import itemgetter, attrgetter
from pyomo.common.config import (
from pyomo.common.gc_manager import PauseGC
from pyomo.common.timing import TicTocTimer
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.label import LPFileLabeler, NumericLabeler
from pyomo.opt import WriterFactory
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.quadratic import QuadraticRepnVisitor
from pyomo.repn.util import (
from pyomo.core.base import Set, RangeSet, ExternalFunction
from pyomo.network import Port
def write_expression(self, ostream, expr, is_objective):
    assert not expr.constant
    getSymbol = self.symbol_map.getSymbol
    getVarOrder = self.var_order.__getitem__
    getVar = self.var_map.__getitem__
    if expr.linear:
        for vid, coef in sorted(expr.linear.items(), key=lambda x: getVarOrder(x[0])):
            if coef < 0:
                ostream.write(f'{coef!r} {getSymbol(getVar(vid))}\n')
            else:
                ostream.write(f'+{coef!r} {getSymbol(getVar(vid))}\n')
    quadratic = getattr(expr, 'quadratic', None)
    if quadratic:

        def _normalize_constraint(data):
            (vid1, vid2), coef = data
            c1 = getVarOrder(vid1)
            c2 = getVarOrder(vid2)
            if c2 < c1:
                col = (c2, c1)
                sym = f' {getSymbol(getVar(vid2))} * {getSymbol(getVar(vid1))}\n'
            elif c1 == c2:
                col = (c1, c1)
                sym = f' {getSymbol(getVar(vid2))} ^ 2\n'
            else:
                col = (c1, c2)
                sym = f' {getSymbol(getVar(vid1))} * {getSymbol(getVar(vid2))}\n'
            if coef < 0:
                return (col, repr(coef) + sym)
            else:
                return (col, '+' + repr(coef) + sym)
        if is_objective:

            def _normalize_objective(data):
                vids, coef = data
                return _normalize_constraint((vids, 2 * coef))
            _normalize = _normalize_objective
        else:
            _normalize = _normalize_constraint
        ostream.write('+ [\n')
        quadratic = sorted(map(_normalize, quadratic.items()), key=itemgetter(0))
        ostream.write(''.join(map(itemgetter(1), quadratic)))
        if is_objective:
            ostream.write('] / 2\n')
        else:
            ostream.write(']\n')