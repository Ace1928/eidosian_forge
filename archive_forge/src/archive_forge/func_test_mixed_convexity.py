import pytest
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms import scalarize
def test_mixed_convexity(self) -> None:
    obj_1 = self.objectives[0]
    obj_2 = cp.Maximize(-self.objectives[1].args[0])
    objectives = [obj_1, obj_2]
    targets = [1, -1]
    priorities = [1, 1]
    with pytest.raises(ValueError, match='Scalarized objective is neither convex nor concave'):
        scalarize.targets_and_priorities(objectives, priorities, targets)
    priorities = [1, -1]
    limits = [1, -1]
    scalarized = scalarize.targets_and_priorities(objectives, priorities, targets, limits)
    assert scalarized.args[0].is_convex()
    priorities = [-1, 1]
    limits = [1, -1]
    scalarized = scalarize.targets_and_priorities(objectives, priorities, targets, limits)
    assert scalarized.args[0].is_concave()