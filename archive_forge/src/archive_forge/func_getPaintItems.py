import os
import re
from ..GraphicsScene import GraphicsScene
from ..Qt import QtCore, QtWidgets
from ..widgets.FileDialog import FileDialog
def getPaintItems(self, root=None):
    """Return a list of all items that should be painted in the correct order."""
    if root is None:
        root = self.item
    preItems = []
    postItems = []
    if isinstance(root, QtWidgets.QGraphicsScene):
        childs = [i for i in root.items() if i.parentItem() is None]
        rootItem = []
    else:
        childs = root.childItems()
        rootItem = [root]
    childs.sort(key=lambda a: a.zValue())
    while len(childs) > 0:
        ch = childs.pop(0)
        tree = self.getPaintItems(ch)
        if ch.flags() & ch.GraphicsItemFlag.ItemStacksBehindParent or (ch.zValue() < 0 and ch.flags() & ch.GraphicsItemFlag.ItemNegativeZStacksBehindParent):
            preItems.extend(tree)
        else:
            postItems.extend(tree)
    return preItems + rootItem + postItems