import sys
from functools import partial
from pydev_ipython.version import check_version
def can_import(api):
    """Safely query whether an API is importable, without importing it"""
    if not has_binding(api):
        return False
    current = loaded_api()
    if api == QT_API_PYQT_DEFAULT:
        return current in [QT_API_PYQT, QT_API_PYQTv1, QT_API_PYQT5, None]
    else:
        return current in [api, None]