import pytest
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms import scalarize
def test_targets_and_priorities(self) -> None:
    targets = [1, 1]
    priorities = [1, 1]
    scalarized = scalarize.targets_and_priorities(self.objectives, priorities, targets)
    prob = cp.Problem(scalarized)
    prob.solve()
    self.assertItemsAlmostEqual(self.x.value, 0.5, places=3)
    targets = [1, 0]
    priorities = [1, 1]
    scalarized = scalarize.targets_and_priorities(self.objectives, priorities, targets)
    prob = cp.Problem(scalarized)
    prob.solve()
    self.assertItemsAlmostEqual(self.x.value, 1, places=3)
    limits = [1, 0.25]
    targets = [0, 0]
    priorities = [1, 0.0001]
    scalarized = scalarize.targets_and_priorities(self.objectives, priorities, targets, limits, off_target=1e-05)
    prob = cp.Problem(scalarized)
    prob.solve()
    self.assertItemsAlmostEqual(self.x.value, 0.5, places=3)
    targets = [-1, 0]
    priorities = [1, 1]
    max_objectives = [cp.Maximize(-obj.args[0]) for obj in self.objectives]
    scalarized = scalarize.targets_and_priorities(max_objectives, priorities, targets, off_target=1e-05)
    assert scalarized.args[0].is_concave()
    prob = cp.Problem(scalarized)
    prob.solve()
    self.assertItemsAlmostEqual(self.x.value, 1, places=3)
    limits = [-1, -0.25]
    targets = [0, 0]
    priorities = [1, 0.0001]
    max_objectives = [cp.Maximize(-obj.args[0]) for obj in self.objectives]
    scalarized = scalarize.targets_and_priorities(max_objectives, priorities, targets, limits, off_target=1e-05)
    assert scalarized.args[0].is_concave()
    prob = cp.Problem(scalarized)
    prob.solve()
    self.assertItemsAlmostEqual(self.x.value, 0.5, places=3)