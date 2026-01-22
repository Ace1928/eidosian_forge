import json
import os
import pyomo.common.unittest as unittest
import pyomo.scripting.pyomo_main as main
from pyomo.opt import check_available_solvers
@unittest.skipUnless(sqlite3_available, 'Requires SQLite3')
def test_pyomo_sqlite3(self):
    results_file = self.run_pyomo(os.path.join(exdir, 'diet1.py'), os.path.join(exdir, 'diet1.sqlite.dat'), outputpath=os.path.join(currdir, 'pyomo_sqlite3.jsn'))
    baseline_file = os.path.join(currdir, 'baselines', 'diet1_pyomo_sqlite3.jsn')
    self.compare_json(results_file, baseline_file)