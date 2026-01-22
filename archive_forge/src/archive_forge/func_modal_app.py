from kivy.tests import async_run, UnitKivyApp
from math import isclose
def modal_app():
    """ test app factory function. """
    from kivy.app import App
    from kivy.uix.button import Button
    from kivy.uix.modalview import ModalView

    class ModalButton(Button):
        """ button used as root widget to test touch. """
        modal = None

        def on_touch_down(self, touch):
            """ touch down event handler. """
            assert self.modal._window is None
            assert not self.modal._is_open
            return super(ModalButton, self).on_touch_down(touch)

        def on_touch_move(self, touch):
            """ touch move event handler. """
            assert self.modal._window is None
            assert not self.modal._is_open
            return super(ModalButton, self).on_touch_move(touch)

        def on_touch_up(self, touch):
            """ touch up event handler. """
            assert self.modal._window is None
            assert not self.modal._is_open
            return super(ModalButton, self).on_touch_up(touch)

    class TestApp(UnitKivyApp, App):
        """ test app class. """

        def build(self):
            """ build root layout. """
            root = ModalButton()
            root.modal = ModalView(size_hint=(0.2, 0.5))
            return root
    return TestApp()