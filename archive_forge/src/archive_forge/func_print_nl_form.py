import logging
from pyomo.core.base import (
from pyomo.mpec.complementarity import Complementarity
from pyomo.gdp import Disjunct
def print_nl_form(self, instance):
    """
        Summarize the complementarity relations in this problem.
        """
    vmap = {}
    for vdata in instance.component_data_objects(Var, active=True):
        vmap[id(vdata)] = vdata
    print('-------------------- Complementary Relations ----------------------')
    for bdata in instance.block_data_objects(active=True, sort=SortComponents.deterministic):
        for cobj in bdata.component_data_objects(Constraint, active=True, descend_into=False):
            print('%s %s\t\t\t%s' % (getattr(cobj, '_complementarity', None), str(cobj.lower) + ' < ' + str(cobj.body) + ' < ' + str(cobj.upper), vmap.get(getattr(cobj, '_vid', None), None)))
    print('-------------------- Complementary Relations ----------------------')