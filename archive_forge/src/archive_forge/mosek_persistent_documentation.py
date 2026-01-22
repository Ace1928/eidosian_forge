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

        Write the model to a file. MOSEK can write files in various
        popular formats such as: lp, mps, ptf, cbf etc.
        In addition to the file formats mentioned above, MOSEK can
        also write files to native formats such as : opf, task and
        jtask. The task format is binary, and is the preferred format
        for sharing with the MOSEK staff in case of queries, since it saves
        the status of the problem and the solver down the smallest detail.
        Parameters:
        filename: str (Name of the output file, including the desired extension)
        