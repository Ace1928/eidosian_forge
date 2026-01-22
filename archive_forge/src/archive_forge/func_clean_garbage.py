from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.lang import Builder
from kivy.base import EventLoop
from kivy.weakproxy import WeakProxy
from time import sleep
def clean_garbage(self, *args):
    for child in self._win.children[:]:
        self._win.remove_widget(child)
    self.move_frames(5)