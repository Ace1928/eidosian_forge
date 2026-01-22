import sys
from random import randrange
from PySide2.QtCore import QAbstractTableModel, QModelIndex, QRect, Qt
from PySide2.QtGui import QColor, QPainter
from PySide2.QtWidgets import (QApplication, QGridLayout, QHeaderView,
from PySide2.QtCharts import QtCharts
PySide2 port of the Model Data example from Qt v5.x