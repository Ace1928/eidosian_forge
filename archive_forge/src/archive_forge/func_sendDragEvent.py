import warnings
import weakref
from time import perf_counter, perf_counter_ns
from .. import debug as debug
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets, isQObjectAlive
from .mouseEvents import HoverEvent, MouseClickEvent, MouseDragEvent
def sendDragEvent(self, ev, init=False, final=False):
    event = MouseDragEvent(ev, self.clickEvents[0], self.lastDrag, start=init, finish=final)
    if init and self.dragItem is None:
        if self.lastHoverEvent is not None:
            acceptedItem = self.lastHoverEvent.dragItems().get(event.button(), None)
        else:
            acceptedItem = None
        if acceptedItem is not None and acceptedItem.scene() is self:
            self.dragItem = acceptedItem
            event.currentItem = self.dragItem
            try:
                self.dragItem.mouseDragEvent(event)
            except:
                debug.printExc('Error sending drag event:')
        else:
            for item in self.itemsNearEvent(event):
                if not item.isVisible() or not item.isEnabled():
                    continue
                if hasattr(item, 'mouseDragEvent'):
                    event.currentItem = item
                    try:
                        item.mouseDragEvent(event)
                    except:
                        debug.printExc('Error sending drag event:')
                    if event.isAccepted():
                        self.dragItem = item
                        if item.flags() & item.GraphicsItemFlag.ItemIsFocusable:
                            item.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
                        break
    elif self.dragItem is not None:
        event.currentItem = self.dragItem
        try:
            self.dragItem.mouseDragEvent(event)
        except:
            debug.printExc('Error sending hover exit event:')
    self.lastDrag = event
    return event.isAccepted()