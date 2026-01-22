from pyomo.contrib.gdpopt.util import get_main_elapsed_time
from pyomo.core import ConstraintList
from pyomo.opt import SolverFactory
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_GOA_config
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.contrib.mindtpy.cut_generation import add_affine_cuts
def update_primal_bound(self, bound_value):
    """Update the primal bound.

        Call after solve fixed NLP subproblem.
        Use the optimal primal bound of the relaxed problem to update the dual bound.

        Parameters
        ----------
        bound_value : float
            The input value used to update the primal bound.
        """
    super().update_primal_bound(bound_value)
    self.primal_bound_progress_time.append(get_main_elapsed_time(self.timing))
    if self.primal_bound_improved:
        self.num_no_good_cuts_added.update({self.primal_bound: len(self.mip.MindtPy_utils.cuts.no_good_cuts)})