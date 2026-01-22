import numpy as np
import skimage.graph.spath as spath
from skimage._shared.testing import assert_equal, assert_array_equal
def test_reach():
    x = np.array([[1, 1, 3], [0, 2, 0], [4, 3, 1]])
    path, cost = spath.shortest_path(x, reach=2)
    assert_array_equal(path, [0, 0, 2])
    assert_equal(cost, 0)