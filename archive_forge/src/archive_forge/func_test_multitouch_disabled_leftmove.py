from kivy.tests.common import GraphicUnitTest
def test_multitouch_disabled_leftmove(self):
    self.multitouch_dot_move('left', disabled=True)