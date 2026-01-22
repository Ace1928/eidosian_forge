import unittest
from kivy.uix.stacklayout import StackLayout
from kivy.uix.widget import Widget
def test_stacklayout_no_children(self):
    sl = StackLayout()
    sl.do_layout()