from __future__ import print_function, absolute_import
import sys
import struct
import os
from shibokensupport.signature import typing
from shibokensupport.signature.typing import TypeVar, Generic
from shibokensupport.signature.lib.tool import with_metaclass
def init_PySide2_QtDataVisualization():
    from PySide2.QtDataVisualization import QtDataVisualization
    QtDataVisualization.QBarDataRow = typing.List[QtDataVisualization.QBarDataItem]
    QtDataVisualization.QBarDataArray = typing.List[QtDataVisualization.QBarDataRow]
    QtDataVisualization.QSurfaceDataRow = typing.List[QtDataVisualization.QSurfaceDataItem]
    QtDataVisualization.QSurfaceDataArray = typing.List[QtDataVisualization.QSurfaceDataRow]
    type_map.update({'100.0f': 100.0})
    return locals()