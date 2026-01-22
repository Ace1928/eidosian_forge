from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.lang import Builder
from kivy.base import EventLoop
from kivy.weakproxy import WeakProxy
from time import sleep
def test_1_openclose(self, *args):
    self._win = EventLoop.window
    self.clean_garbage()
    root = Builder.load_string(KV)
    self.render(root)
    self.assertLess(len(self._win.children), 2)
    group2 = root.ids.group2
    group1 = root.ids.group1
    self.move_frames(5)
    self.check_dropdown(present=False)
    self.assertFalse(group2.is_open)
    self.assertFalse(group1.is_open)
    items = ((group2, group1), (group1, group2))
    for item in items:
        active, passive = item
        TouchPoint(*active.center)
        self.check_dropdown(present=True)
        gdd = WeakProxy(self._win.children[0])
        self.assertIn(gdd, self._win.children)
        self.assertEqual(gdd, self._win.children[0])
        self.assertTrue(active.is_open)
        self.assertFalse(passive.is_open)
        TouchPoint(0, 0)
        sleep(gdd.min_state_time)
        self.move_frames(1)
        self.assertNotEqual(gdd, self._win.children[0])
        self.assertLess(len(self._win.children), 2)
        self.check_dropdown(present=False)
        self.assertFalse(active.is_open)
        self.assertFalse(passive.is_open)
    self._win.remove_widget(root)