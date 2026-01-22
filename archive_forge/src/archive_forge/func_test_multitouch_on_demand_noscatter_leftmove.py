from kivy.tests.common import GraphicUnitTest
def test_multitouch_on_demand_noscatter_leftmove(self):
    self.multitouch_dot_move('left', on_demand=True)