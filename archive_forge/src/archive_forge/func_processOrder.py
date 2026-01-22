import importlib
import os
from collections import OrderedDict
from numpy import ndarray
from .. import DataTreeWidget, FileDialog
from .. import configfile as configfile
from .. import dockarea as dockarea
from .. import functions as fn
from ..debug import printExc
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Qt import QtCore, QtWidgets
from . import FlowchartCtrlTemplate_generic as FlowchartCtrlTemplate
from . import FlowchartGraphicsView
from .library import LIBRARY
from .Node import Node
from .Terminal import Terminal
def processOrder(self):
    """Return the order of operations required to process this chart.
        The order returned should look like [('p', node1), ('p', node2), ('d', terminal1), ...] 
        where each tuple specifies either (p)rocess this node or (d)elete the result from this terminal
        """
    deps = {}
    tdeps = {}
    for name, node in self._nodes.items():
        deps[node] = node.dependentNodes()
        for t in node.outputs().values():
            tdeps[t] = t.dependentNodes()
    order = fn.toposort(deps)
    ops = [('p', n) for n in order]
    dels = []
    for t, nodes in tdeps.items():
        lastInd = 0
        lastNode = None
        for n in nodes:
            if n is self:
                lastInd = None
                break
            else:
                try:
                    ind = order.index(n)
                except ValueError:
                    continue
            if lastNode is None or ind > lastInd:
                lastNode = n
                lastInd = ind
        if lastInd is not None:
            dels.append((lastInd + 1, t))
    dels.sort(key=lambda a: a[0], reverse=True)
    for i, t in dels:
        ops.insert(i, ('d', t))
    return ops