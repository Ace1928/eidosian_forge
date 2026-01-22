import numpy as np
import scipy  # For version checks
import cvxpy.settings as s
from cvxpy.reductions.solvers.conic_solvers.cbc_conif import CBC as CBC_con
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL as CLARABEL_con
from cvxpy.reductions.solvers.conic_solvers.copt_conif import COPT as COPT_con
from cvxpy.reductions.solvers.conic_solvers.cplex_conif import CPLEX as CPLEX_con
from cvxpy.reductions.solvers.conic_solvers.cvxopt_conif import CVXOPT as CVXOPT_con
from cvxpy.reductions.solvers.conic_solvers.diffcp_conif import DIFFCP as DIFFCP_con
from cvxpy.reductions.solvers.conic_solvers.ecos_bb_conif import ECOS_BB as ECOS_BB_con
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import ECOS as ECOS_con
from cvxpy.reductions.solvers.conic_solvers.glop_conif import GLOP as GLOP_con
from cvxpy.reductions.solvers.conic_solvers.glpk_conif import GLPK as GLPK_con
from cvxpy.reductions.solvers.conic_solvers.glpk_mi_conif import GLPK_MI as GLPK_MI_con
from cvxpy.reductions.solvers.conic_solvers.gurobi_conif import GUROBI as GUROBI_con
from cvxpy.reductions.solvers.conic_solvers.mosek_conif import MOSEK as MOSEK_con
from cvxpy.reductions.solvers.conic_solvers.nag_conif import NAG as NAG_con
from cvxpy.reductions.solvers.conic_solvers.pdlp_conif import PDLP as PDLP_con
from cvxpy.reductions.solvers.conic_solvers.scip_conif import SCIP as SCIP_con
from cvxpy.reductions.solvers.conic_solvers.scipy_conif import SCIPY as SCIPY_con
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS as SCS_con
from cvxpy.reductions.solvers.conic_solvers.sdpa_conif import SDPA as SDPA_con
from cvxpy.reductions.solvers.conic_solvers.xpress_conif import XPRESS as XPRESS_con
from cvxpy.reductions.solvers.qp_solvers.copt_qpif import COPT as COPT_qp
from cvxpy.reductions.solvers.qp_solvers.cplex_qpif import CPLEX as CPLEX_qp
from cvxpy.reductions.solvers.qp_solvers.gurobi_qpif import GUROBI as GUROBI_qp
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP as OSQP_qp
from cvxpy.reductions.solvers.qp_solvers.piqp_qpif import PIQP as PIQP_qp
from cvxpy.reductions.solvers.qp_solvers.proxqp_qpif import PROXQP as PROXQP_qp
from cvxpy.reductions.solvers.qp_solvers.xpress_qpif import XPRESS as XPRESS_qp
from cvxpy.utilities.versioning import Version
def installed_solvers():
    """List the installed solvers.
    """
    installed = []
    for name, solver in SOLVER_MAP_CONIC.items():
        if solver.is_installed():
            installed.append(name)
    for name, solver in SOLVER_MAP_QP.items():
        if solver.is_installed():
            installed.append(name)
    return np.unique(installed).tolist()