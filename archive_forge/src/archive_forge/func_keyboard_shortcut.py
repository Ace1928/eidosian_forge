import kivy
import weakref
from functools import partial
from itertools import chain
from kivy.logger import Logger
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.treeview import TreeViewNode, TreeView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.modalview import ModalView
from kivy.graphics import Color, Rectangle, PushMatrix, PopMatrix
from kivy.graphics.context_instructions import Transform
from kivy.graphics.transformation import Matrix
from kivy.properties import (ObjectProperty, BooleanProperty, ListProperty,
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.lang import Builder
def keyboard_shortcut(self, win, scancode, *largs):
    modifiers = largs[-1]
    if scancode == 101 and modifiers == ['ctrl']:
        self.activated = not self.activated
        if self.activated:
            self.inspect_enabled = True
        return True
    elif scancode == 27:
        if self.inspect_enabled:
            self.inspect_enabled = False
            return True
        if self.activated:
            self.activated = False
            return True
    if not self.activated or not self.widget:
        return
    if scancode == 273:
        self.widget = self.widget.parent
    elif scancode == 274:
        filtered_children = [c for c in self.widget.children if not isinstance(c, Console)]
        if filtered_children:
            self.widget = filtered_children[0]
    elif scancode == 276:
        parent = self.widget.parent
        filtered_children = [c for c in parent.children if not isinstance(c, Console)]
        index = filtered_children.index(self.widget)
        index = max(0, index - 1)
        self.widget = filtered_children[index]
    elif scancode == 275:
        parent = self.widget.parent
        filtered_children = [c for c in parent.children if not isinstance(c, Console)]
        index = filtered_children.index(self.widget)
        index = min(len(filtered_children) - 1, index + 1)
        self.widget = filtered_children[index]