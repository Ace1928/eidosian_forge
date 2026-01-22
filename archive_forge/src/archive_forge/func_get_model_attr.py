from collections.abc import Iterable
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect, gurobipy
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.numvalue import value, is_fixed
from pyomo.opt.base import SolverFactory
def get_model_attr(self, attr):
    """Get the value of an attribute on the Gurobi model.

        Parameters
        ----------
        attr: str
            The attribute to get. See Gurobi documentation for
            descriptions of the attributes.

            Options are:

                NumVars
                NumConstrs
                NumSOS
                NumQConstrs
                NumgGenConstrs
                NumNZs
                DNumNZs
                NumQNZs
                NumQCNZs
                NumIntVars
                NumBinVars
                NumPWLObjVars
                ModelName
                ModelSense
                ObjCon
                ObjVal
                ObjBound
                ObjBoundC
                PoolObjBound
                PoolObjVal
                MIPGap
                Runtime
                Status
                SolCount
                IterCount
                BarIterCount
                NodeCount
                IsMIP
                IsQP
                IsQCP
                IsMultiObj
                IISMinimal
                MaxCoeff
                MinCoeff
                MaxBound
                MinBound
                MaxObjCoeff
                MinObjCoeff
                MaxRHS
                MinRHS
                MaxQCCoeff
                MinQCCoeff
                MaxQCLCoeff
                MinQCLCoeff
                MaxQCRHS
                MinQCRHS
                MaxQObjCoeff
                MinQObjCoeff
                Kappa
                KappaExact
                FarkasProof
                TuneResultCount
                LicenseExpiration
                BoundVio
                BoundSVio
                BoundVioIndex
                BoundSVioIndex
                BoundVioSum
                BoundSVioSum
                ConstrVio
                ConstrSVio
                ConstrVioIndex
                ConstrSVioIndex
                ConstrVioSum
                ConstrSVioSum
                ConstrResidual
                ConstrSResidual
                ConstrResidualIndex
                ConstrSResidualIndex
                ConstrResidualSum
                ConstrSResidualSum
                DualVio
                DualSVio
                DualVioIndex
                DualSVioIndex
                DualVioSum
                DualSVioSum
                DualResidual
                DualSResidual
                DualResidualIndex
                DualSResidualIndex
                DualResidualSum
                DualSResidualSum
                ComplVio
                ComplVioIndex
                ComplVioSum
                IntVio
                IntVioIndex
                IntVioSum

        """
    if self._needs_updated:
        self._update()
    return self._solver_model.getAttr(attr)