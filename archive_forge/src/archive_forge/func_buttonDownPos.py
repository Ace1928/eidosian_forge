import weakref
from time import perf_counter
from ..Point import Point
from ..Qt import QtCore
def buttonDownPos(self, btn=None):
    """
        Return the position of the mouse at the time the drag was initiated
        in the coordinate system of the item that the event was delivered to.
        """
    if btn is None:
        btn = self.button()
    return Point(self.currentItem.mapFromScene(self._buttonDownScenePos[btn]))