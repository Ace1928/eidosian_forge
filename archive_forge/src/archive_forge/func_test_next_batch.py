from heat.tests import common
from heat.scaling import rolling_update
def test_next_batch(self):
    batch = rolling_update.next_batch(self.targ, self.curr, self.updated, self.bat_size, self.min_srv)
    self.assertEqual(self.batch, batch)