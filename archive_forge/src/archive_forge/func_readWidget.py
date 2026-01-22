import inspect
import weakref
from .Qt import QtCore, QtWidgets
def readWidget(self, w):
    if type(w) in WidgetGroup.classes:
        getFunc = WidgetGroup.classes[type(w)][1]
    else:
        getFunc = w.widgetGroupInterface()[1]
    if getFunc is None:
        return None
    if inspect.ismethod(getFunc) and getFunc.__self__ is not None:
        val = getFunc()
    else:
        val = getFunc(w)
    if self.scales[w] is not None:
        val /= self.scales[w]
    n = self.widgetList[w]
    self.cache[n] = val
    return val