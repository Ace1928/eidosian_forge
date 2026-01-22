import sys
from functools import partial
from pydev_ipython.version import check_version
def has_binding(api):
    """Safely check for PyQt4 or PySide, without importing
       submodules

       Parameters
       ----------
       api : str [ 'pyqtv1' | 'pyqt' | 'pyside' | 'pyqtdefault']
            Which module to check for

       Returns
       -------
       True if the relevant module appears to be importable
    """
    module_name = {QT_API_PYSIDE: 'PySide', QT_API_PYSIDE2: 'PySide2', QT_API_PYQT: 'PyQt4', QT_API_PYQTv1: 'PyQt4', QT_API_PYQT_DEFAULT: 'PyQt4', QT_API_PYQT5: 'PyQt5'}
    module_name = module_name[api]
    import imp
    try:
        mod = __import__(module_name)
        imp.find_module('QtCore', mod.__path__)
        imp.find_module('QtGui', mod.__path__)
        imp.find_module('QtSvg', mod.__path__)
        if api == QT_API_PYSIDE:
            return check_version(mod.__version__, '1.0.3')
        else:
            return True
    except ImportError:
        return False