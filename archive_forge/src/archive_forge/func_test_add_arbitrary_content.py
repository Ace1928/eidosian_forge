import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.base import EventLoop
from kivy.uix.bubble import Bubble
from kivy.uix.bubble import BubbleContent
from kivy.uix.bubble import BubbleButton
def test_add_arbitrary_content(self):
    from kivy.uix.gridlayout import GridLayout
    bubble = Bubble()
    content = GridLayout()
    bubble.add_widget(content)
    self.render(bubble)