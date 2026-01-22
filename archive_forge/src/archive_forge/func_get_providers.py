from kivy.tests.common import GraphicUnitTest
def get_providers(self, with_window_children=False):
    from kivy.base import EventLoop
    win = EventLoop.window
    if with_window_children:
        from kivy.uix.button import Button
        button = Button(on_touch_down=self.on_any_touch_event, on_touch_move=self.on_any_touch_event, on_touch_up=self.on_any_touch_event)
        self.button_widget = button
        win.add_widget(button)
    return (win, self.mouse)