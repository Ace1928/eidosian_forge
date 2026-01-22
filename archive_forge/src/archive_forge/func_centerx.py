import doctest
import collections
@centerx.setter
def centerx(self, newCenterx):
    if self._readOnly:
        raise PyRectException('Rect object is read-only')
    _checkForIntOrFloat(newCenterx)
    originalLeft = self._left
    if self._enableFloat:
        if newCenterx != self._left + self._width / 2.0:
            self._left = newCenterx - self._width / 2.0
            self.callOnChange(originalLeft, self._top, self._width, self._height)
    elif newCenterx != self._left + self._width // 2:
        self._left = int(newCenterx) - self._width // 2
        self.callOnChange(originalLeft, self._top, self._width, self._height)