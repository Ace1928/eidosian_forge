from PySide2.QtWidgets import (QItemDelegate, QStyledItemDelegate, QStyle)
from starrating import StarRating
from stareditor import StarEditor
def setModelData(self, editor, model, index):
    """ Get the data from our custom editor and stuffs it into the model.
        """
    if index.column() == 3:
        model.setData(index, editor.starRating.starCount)
    else:
        QStyledItemDelegate.setModelData(self, editor, model, index)