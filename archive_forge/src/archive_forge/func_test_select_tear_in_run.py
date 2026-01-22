import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available, networkx_available
from pyomo.environ import (
from pyomo.network import Port, SequentialDecomposition, Arc
from pyomo.gdp.tests.models import makeExpandedNetworkDisjunction
from types import MethodType
import_available = numpy_available and networkx_available
@unittest.skipIf(not glpk_available, 'GLPK solver not available')
def test_select_tear_in_run(self):
    m = self.simple_recycle_model()

    def function(unit):
        unit.initialize()
    seq = SequentialDecomposition()
    tset = [m.stream_splitter_to_mixer]
    seq.set_tear_set(tset)
    splitter_to_mixer_guess = {'flow': {'A': 0, 'B': 0, 'C': 0}, 'temperature': 450, 'pressure': 128}
    seq.set_guesses_for(m.mixer.inlet_side_2, splitter_to_mixer_guess)
    m.mixer.expr_var_idx_in_side_2['A'] = 0
    m.mixer.expr_var_idx_in_side_2['B'] = 0
    m.mixer.expr_var_idx_in_side_2['C'] = 0
    m.mixer.expr_var_in_side_2 = 0
    seq.run(m, function)
    seq = SequentialDecomposition(tear_solver='glpk', select_tear_method='mip')
    seq.run(m, function)
    self.check_recycle_model(m)
    seq = SequentialDecomposition(tear_solver='glpk', select_tear_method='heuristic')
    seq.run(m, function)
    self.check_recycle_model(m)