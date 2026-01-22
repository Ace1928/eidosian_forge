import operator
import itertools
import pyomo.core.base.var
import pyomo.core.base.constraint
from pyomo.core import is_fixed, value
from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.solvers.plugins.solvers.mosek_direct import MOSEKDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.solvers.plugins.solvers.direct_solver import DirectSolver
from pyomo.solvers.plugins.solvers.direct_or_persistent_solver import (
from pyomo.opt.base import SolverFactory
from pyomo.core.kernel.conic import _ConicBase
from pyomo.core.kernel.block import block
def remove_constraints(self, *solver_cons):
    """
        Remove multiple constraints from the model as well as the MOSEK task in one
        method call.

        This will keep any other model components intact.
        To remove conic-domains, use the remove_block method.

        Parameters
        ----------
        *solver_cons: Constraint (scalar Constraint or single _ConstraintData)
        """
    lq_cons = tuple(itertools.filterfalse(lambda x: isinstance(x, _ConicBase), solver_cons))
    cone_cons = tuple(filter(lambda x: isinstance(x, _ConicBase), solver_cons))
    try:
        lq = []
        cones = []
        for c in lq_cons:
            lq.append(self._pyomo_con_to_solver_con_map[c])
            self._symbol_map.removeSymbol(c)
            del self._pyomo_con_to_solver_con_map[c]
        for c in cone_cons:
            cones.append(self._pyomo_cone_to_solver_cone_map[c])
            self._symbol_map.removeSymbol(c)
            del self._pyomo_cone_to_solver_cone_map[c]
        self._solver_model.removecons(lq)
        self._solver_model.removecones(cones)
        lq_num = self._solver_model.getnumcon()
        cone_num = self._solver_model.getnumcone()
    except KeyError:
        c_name = self._symbol_map.getSymbol(c, self._labeler)
        raise ValueError('Constraint/Cone {} needs to be added before removal.'.format(c_name))
    self._solver_con_to_pyomo_con_map = dict(zip(range(lq_num), self._pyomo_con_to_solver_con_map.keys()))
    self._solver_cone_to_pyomo_cone_map = dict(zip(range(cone_num), self._pyomo_cone_to_solver_cone_map.keys()))
    for i, c in enumerate(self._pyomo_con_to_solver_con_map):
        self._pyomo_con_to_solver_con_map[c] = i
    for i, c in enumerate(self._pyomo_cone_to_solver_cone_map):
        self._pyomo_cone_to_solver_cone_map[c] = i