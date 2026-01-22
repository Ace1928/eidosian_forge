from PySide2.QtWidgets import (QWidget)
from PySide2.QtGui import (QPainter)
from PySide2.QtCore import Signal
def starAtPosition(self, x):
    """ Calculate which star the user's mouse cursor is currently
            hovering over.
        """
    star = x / (self.starRating.sizeHint().width() / self.starRating.maxStarCount) + 1
    if star <= 0 or star > self.starRating.maxStarCount:
        return -1
    return star