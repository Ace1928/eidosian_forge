import weakref
from ..Qt import QtWidgets
from .Container import Container, HContainer, TContainer, VContainer
from .Dock import Dock
from .DockDrop import DockDrop
def childState(self, obj):
    if isinstance(obj, Dock):
        return ('dock', obj.name(), {})
    else:
        childs = []
        for i in range(obj.count()):
            childs.append(self.childState(obj.widget(i)))
        return (obj.type(), childs, obj.saveState())