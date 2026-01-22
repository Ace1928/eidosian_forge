from pythran.analyses import Identifiers
from pythran.passmanager import Transformation
import gast as ast
from functools import reduce
from collections import OrderedDict
from copy import deepcopy
def visit_AnyComp(self, node, *fields):
    for field in fields:
        setattr(node, field, self.visit(getattr(node, field)))
    generators = [self.visit(generator) for generator in node.generators]
    nnode = node
    for i, g in enumerate(generators):
        if isinstance(g, tuple):
            gtarget = '{0}{1}'.format(g[0], i)
            nnode.generators[i].target = ast.Name(gtarget, nnode.generators[i].target.ctx, None, None)
            nnode = ConvertToTuple(gtarget, g[1]).visit(nnode)
            self.update = True
    for field in fields:
        setattr(node, field, getattr(nnode, field))
    node.generators = nnode.generators
    return node