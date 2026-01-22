from heat.tests import common
from heat.scaling import rolling_update
def test_needs_update(self):
    needs_update = rolling_update.needs_update(self.targ, self.curr, self.updated)
    self.assertEqual(self.result, needs_update)