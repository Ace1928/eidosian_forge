from kivy.tests.common import GraphicUnitTest
def test_end_event_no_dispatch_through_on_touch_events(self):
    win, mouse = self.get_providers(with_window_children=True)
    win.dispatch('on_cursor_enter')
    x, y = win.mouse_pos = (10.0, 10.0)
    win.dispatch('on_cursor_leave')
    self.advance_frames(1)
    self.assert_event('end', win.to_normalized_pos(x, y))
    assert self.touch_event is None