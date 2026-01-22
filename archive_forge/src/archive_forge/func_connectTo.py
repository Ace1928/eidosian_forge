import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
def connectTo(self, term, connectionItem=None):
    try:
        if self.connectedTo(term):
            raise Exception('Already connected')
        if term is self:
            raise Exception('Not connecting terminal to self')
        if term.node() is self.node():
            raise Exception("Can't connect to terminal on same node.")
        for t in [self, term]:
            if t.isInput() and (not t._multi) and (len(t.connections()) > 0):
                raise Exception('Cannot connect %s <-> %s: Terminal %s is already connected to %s (and does not allow multiple connections)' % (self, term, t, list(t.connections().keys())))
    except:
        if connectionItem is not None:
            connectionItem.close()
        raise
    if connectionItem is None:
        connectionItem = ConnectionItem(self.graphicsItem(), term.graphicsItem())
        self.graphicsItem().getViewBox().addItem(connectionItem)
    self._connections[term] = connectionItem
    term._connections[self] = connectionItem
    self.recolor()
    self.connected(term)
    term.connected(self)
    return connectionItem