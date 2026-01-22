from PySide2.QtWidgets import (QItemDelegate, QStyledItemDelegate, QStyle)
from starrating import StarRating
from stareditor import StarEditor
def setEditorData(self, editor, index):
    """ Sets the data to be displayed and edited by our custom editor. """
    if index.column() == 3:
        editor.starRating = StarRating(index.data())
    else:
        QStyledItemDelegate.setEditorData(self, editor, index)