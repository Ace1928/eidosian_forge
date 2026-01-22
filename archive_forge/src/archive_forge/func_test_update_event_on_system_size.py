from kivy.tests.common import GraphicUnitTest
def test_update_event_on_system_size(self):
    win, mouse = self.get_providers()
    x, y = win.mouse_pos = (10.0, 10.0)
    w, h = win.system_size
    win.system_size = (w + 10, h + 10)
    self.advance_frames(1)
    self.assert_event('update', win.to_normalized_pos(x, y))
    win.dispatch('on_cursor_leave')
    self.advance_frames(1)