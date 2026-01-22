import warnings
import weakref
from time import perf_counter, perf_counter_ns
from .. import debug as debug
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets, isQObjectAlive
from .mouseEvents import HoverEvent, MouseClickEvent, MouseDragEvent
def itemsNearEvent(self, event, selMode=QtCore.Qt.ItemSelectionMode.IntersectsItemShape, sortOrder=QtCore.Qt.SortOrder.DescendingOrder, hoverable=False):
    """
        Return an iterator that iterates first through the items that directly intersect point (in Z order)
        followed by any other items that are within the scene's click radius.
        """
    view = self.views()[0]
    tr = view.viewportTransform()
    if hasattr(event, 'buttonDownScenePos'):
        point = event.buttonDownScenePos()
    else:
        point = event.scenePos()

    def absZValue(item):
        if item is None:
            return 0
        return item.zValue() + absZValue(item.parentItem())
    items_at_point = self.items(point, selMode, sortOrder, tr)
    items_at_point.sort(key=absZValue, reverse=True)
    r = self._clickRadius
    items_within_radius = []
    rgn = None
    if r > 0:
        rect = view.mapToScene(QtCore.QRect(0, 0, 2 * r, 2 * r)).boundingRect()
        w = rect.width()
        h = rect.height()
        rgn = QtCore.QRectF(point.x() - w / 2, point.y() - h / 2, w, h)
        items_within_radius = self.items(rgn, selMode, sortOrder, tr)
        items_within_radius.sort(key=absZValue, reverse=True)
        for item in items_at_point:
            if item in items_within_radius:
                items_within_radius.remove(item)
    all_items = items_at_point + items_within_radius
    selected_items = []
    for item in all_items:
        if hoverable and (not hasattr(item, 'hoverEvent')):
            continue
        if item.scene() is not self:
            continue
        shape = item.shape()
        if shape is None:
            continue
        if rgn is not None and shape.intersects(item.mapFromScene(rgn).boundingRect()) or shape.contains(item.mapFromScene(point)):
            selected_items.append(item)
    return selected_items