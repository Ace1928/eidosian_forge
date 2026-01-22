import doctest
import collections
@bottomleft.setter
def bottomleft(self, value):
    if self._readOnly:
        raise PyRectException('Rect object is read-only')
    _checkForTwoIntOrFloatTuple(value)
    newLeft, newBottom = value
    if newLeft != self._left or newBottom != self._top + self._height:
        originalLeft = self._left
        originalTop = self._top
        if self._enableFloat:
            self._left = newLeft
            self._top = newBottom - self._height
        else:
            self._left = int(newLeft)
            self._top = int(newBottom) - self._height
        self.callOnChange(originalLeft, originalTop, self._width, self._height)