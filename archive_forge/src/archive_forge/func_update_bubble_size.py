import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.base import EventLoop
from kivy.uix.bubble import Bubble
from kivy.uix.bubble import BubbleContent
from kivy.uix.bubble import BubbleButton
def update_bubble_size(instance, value):
    w = bubble_width
    h = bubble.content_height + bubble.arrow_margin_y
    bubble.size = (w, h)