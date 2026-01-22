import copy
from PySide2.QtSql import QSqlRelationalDelegate
from PySide2.QtWidgets import (QItemDelegate, QSpinBox, QStyledItemDelegate,
from PySide2.QtGui import QMouseEvent, QPixmap, QPalette
from PySide2.QtCore import QEvent, QSize, Qt
def editorEvent(self, event, model, option, index):
    if index.column() != 5:
        return False
    if event.type() == QEvent.MouseButtonPress:
        mouse_pos = event.pos()
        new_stars = int(0.7 + (mouse_pos.x() - option.rect.x()) / self.star.width())
        stars = max(0, min(new_stars, 5))
        model.setData(index, stars)
        return False
    return True