import unittest
from kivy.tests.common import GraphicUnitTest, UnitTestTouch
from kivy.base import EventLoop
from kivy.modules import inspector
from kivy.factory import Factory
@unittest.skip("doesn't work on CI with Python 3.5 but works locally")
def test_widget_multipopup(self, *args):
    EventLoop.ensure_window()
    self._win = EventLoop.window
    self.clean_garbage()
    self.root = self.builder.Builder.load_string(KV, filename='InspectorTestCase.KV')
    self.render(self.root)
    self.assertLess(len(self._win.children), 2)
    popup = self.root.ids.popup
    inspector.start(self._win, self.root)
    self.advance_frames(1)
    ins = self.root.inspector
    ins.inspect_enabled = False
    ins.activated = True
    self.assertTrue(ins.at_bottom)
    touch = UnitTestTouch(*popup.center)
    touch.touch_down()
    touch.touch_up()
    self.advance_frames(1)
    touch = UnitTestTouch(self._win.width / 2.0, self._win.height / 2.0)
    for i in range(2):
        touch.touch_down()
        touch.touch_up()
        self.advance_frames(1)
    modals = [Factory.ThirdModal, Factory.SecondModal, Factory.FirstModal]
    for mod in modals:
        ins.inspect_enabled = True
        self.advance_frames(1)
        touch.touch_down()
        touch.touch_up()
        self.advance_frames(1)
        self.assertIsInstance(ins.widget, Factory.Button)
        self.assertIsInstance(ins.widget.parent, mod)
        ins.inspect_enabled = False
        orig = UnitTestTouch(0, 0)
        orig.touch_down()
        orig.touch_up()
        self.advance_frames(10)
    ins.activated = False
    self.render(self.root)
    self.advance_frames(5)
    inspector.stop(self._win, self.root)
    self.assertLess(len(self._win.children), 2)
    self.render(self.root)