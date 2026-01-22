import doctest
import collections
@midleft.setter
def midleft(self, value):
    if self._readOnly:
        raise PyRectException('Rect object is read-only')
    _checkForTwoIntOrFloatTuple(value)
    newLeft, newMidLeft = value
    originalLeft = self._left
    originalTop = self._top
    if self._enableFloat:
        if newLeft != self._left or newMidLeft != self._top + self._height / 2.0:
            self._left = newLeft
            self._top = newMidLeft - self._height / 2.0
            self.callOnChange(originalLeft, originalTop, self._width, self._height)
    elif newLeft != self._left or newMidLeft != self._top + self._height // 2:
        self._left = int(newLeft)
        self._top = int(newMidLeft) - self._height // 2
        self.callOnChange(originalLeft, originalTop, self._width, self._height)