import json
import os
import pyomo.common.unittest as unittest
import pyomo.scripting.pyomo_main as main
from pyomo.opt import check_available_solvers
@unittest.skipUnless(pyodbc_available, 'Requires PyODBC')
def test_pyomo_mdb(self):
    results_file = self.run_pyomo(os.path.join(exdir, 'diet1.py'), os.path.join(exdir, 'diet1.db.dat'), outputpath=os.path.join(currdir, 'pyomo_mdb.jsn'))
    baseline_file = os.path.join(currdir, 'baselines', 'diet1_pyomo_mdb.jsn')
    self.compare_json(results_file, baseline_file)