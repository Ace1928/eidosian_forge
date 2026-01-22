from kivy.tests.common import GraphicUnitTest
def test_no_event_on_system_size(self):
    win, mouse = self.get_providers()
    w, h = win.system_size
    win.system_size = (w + 10, h + 10)
    self.advance_frames(1)
    self.assert_no_event()