import doctest
import collections
@midbottom.setter
def midbottom(self, value):
    if self._readOnly:
        raise PyRectException('Rect object is read-only')
    _checkForTwoIntOrFloatTuple(value)
    newMidBottom, newBottom = value
    originalLeft = self._left
    originalTop = self._top
    if self._enableFloat:
        if newMidBottom != self._left + self._width / 2.0 or newBottom != self._top + self._height:
            self._left = newMidBottom - self._width / 2.0
            self._top = newBottom - self._height
            self.callOnChange(originalLeft, originalTop, self._width, self._height)
    elif newMidBottom != self._left + self._width // 2 or newBottom != self._top + self._height:
        self._left = int(newMidBottom) - self._width // 2
        self._top = int(newBottom) - self._height
        self.callOnChange(originalLeft, originalTop, self._width, self._height)