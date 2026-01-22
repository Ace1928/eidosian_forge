from __future__ import print_function
import os
import sys
from PySide2.QtCore import QObject, QUrl
from PySide2.QtGui import QGuiApplication
import PySide2.QtQml
from PySide2.QtQuick import QQuickView
def sayThis(s):
    print(s)