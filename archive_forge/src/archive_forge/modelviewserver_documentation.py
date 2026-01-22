import sys
from PySide2.QtCore import (Qt, QByteArray, QModelIndex, QObject, QTimer, QUrl)
from PySide2.QtGui import (QColor, QStandardItemModel, QStandardItem)
from PySide2.QtWidgets import (QApplication, QTreeView)
from PySide2.QtRemoteObjects import (QRemoteObjectHost, QRemoteObjectNode,
PySide2 port of the remoteobjects/modelviewserver example from Qt v5.x