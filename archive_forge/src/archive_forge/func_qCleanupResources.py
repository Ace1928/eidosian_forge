from PySide2 import QtCore
def qCleanupResources():
    QtCore.qUnregisterResourceData(1, qt_resource_struct, qt_resource_name, qt_resource_data)