from taskflow import deciders
from taskflow import test
def test_pick_widest(self):
    choices = [deciders.Depth.ATOM, deciders.Depth.FLOW]
    self.assertEqual(deciders.Depth.FLOW, deciders.pick_widest(choices))
    choices = [deciders.Depth.ATOM, deciders.Depth.FLOW, deciders.Depth.ALL]
    self.assertEqual(deciders.Depth.ALL, deciders.pick_widest(choices))
    choices = [deciders.Depth.ATOM, deciders.Depth.FLOW, deciders.Depth.ALL, deciders.Depth.NEIGHBORS]
    self.assertEqual(deciders.Depth.ALL, deciders.pick_widest(choices))
    choices = [deciders.Depth.ATOM, deciders.Depth.NEIGHBORS]
    self.assertEqual(deciders.Depth.NEIGHBORS, deciders.pick_widest(choices))