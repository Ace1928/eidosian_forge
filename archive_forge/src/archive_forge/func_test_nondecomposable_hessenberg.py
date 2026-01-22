from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.common.dependencies import scipy, scipy_available, networkx_available
import pyomo.common.unittest as unittest
def test_nondecomposable_hessenberg(self):
    """
        |x x      |
        |  x x    |
        |    x x  |
        |      x x|
        |x x x x x|
        """
    N = 5
    row = []
    col = []
    data = []
    for i in range(N):
        row.append(N - 1)
        col.append(i)
        data.append(1)
        row.append(i)
        col.append(i)
        data.append(1)
        if i != 0:
            row.append(i - 1)
            col.append(i)
            data.append(1)
    matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))
    matching = maximum_matching(matrix)
    values = set(matching.values())
    self.assertEqual(len(matching), N)
    for i in range(N):
        self.assertIn(i, matching)
        self.assertIn(i, values)