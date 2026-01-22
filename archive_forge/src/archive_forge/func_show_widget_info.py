import weakref
from functools import partial
from itertools import chain
from kivy.animation import Animation
from kivy.logger import Logger
from kivy.graphics.transformation import Matrix
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.weakproxy import WeakProxy
from kivy.properties import (
def show_widget_info(self):
    self.content.clear_widgets()
    widget = self.widget
    treeview = self.treeview
    for node in list(treeview.iterate_all_nodes())[:]:
        node.widget_ref = None
        treeview.remove_node(node)
    if not widget:
        if self.at_bottom:
            Animation(top=60, t='out_quad', d=0.3).start(self.layout)
        else:
            Animation(y=self.height - 60, t='out_quad', d=0.3).start(self.layout)
        self.widget_info = False
        return
    self.widget_info = True
    if self.at_bottom:
        Animation(top=250, t='out_quad', d=0.3).start(self.layout)
    else:
        Animation(top=self.height, t='out_quad', d=0.3).start(self.layout)
    for node in list(treeview.iterate_all_nodes())[:]:
        treeview.remove_node(node)
    keys = list(widget.properties().keys())
    keys.sort()
    node = None
    if type(widget) is WeakProxy:
        wk_widget = widget.__ref__
    else:
        wk_widget = weakref.ref(widget)
    for key in keys:
        node = TreeViewProperty(key=key, widget_ref=wk_widget)
        node.bind(is_selected=self.show_property)
        try:
            widget.bind(**{key: partial(self.update_node_content, weakref.ref(node))})
        except:
            pass
        treeview.add_node(node)