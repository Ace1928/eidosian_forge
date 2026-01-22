from __future__ import nested_scopes
from _pydev_bundle._pydev_saved_modules import threading
import os
from _pydev_bundle import pydev_log
def patch_qt(qt_support_mode):
    """
    This method patches qt (PySide2, PySide, PyQt4, PyQt5) so that we have hooks to set the tracing for QThread.
    """
    if not qt_support_mode:
        return
    if qt_support_mode is True or qt_support_mode == 'True':
        qt_support_mode = 'auto'
    if qt_support_mode == 'auto':
        qt_support_mode = os.getenv('PYDEVD_PYQT_MODE', 'auto')
    global _patched_qt
    if _patched_qt:
        return
    pydev_log.debug('Qt support mode: %s', qt_support_mode)
    _patched_qt = True
    if qt_support_mode == 'auto':
        patch_qt_on_import = None
        try:
            import PySide2
            qt_support_mode = 'pyside2'
        except:
            try:
                import Pyside
                qt_support_mode = 'pyside'
            except:
                try:
                    import PyQt5
                    qt_support_mode = 'pyqt5'
                except:
                    try:
                        import PyQt4
                        qt_support_mode = 'pyqt4'
                    except:
                        return
    if qt_support_mode == 'pyside2':
        try:
            import PySide2.QtCore
            _internal_patch_qt(PySide2.QtCore, qt_support_mode)
        except:
            return
    elif qt_support_mode == 'pyside':
        try:
            import PySide.QtCore
            _internal_patch_qt(PySide.QtCore, qt_support_mode)
        except:
            return
    elif qt_support_mode == 'pyqt5':
        try:
            import PyQt5.QtCore
            _internal_patch_qt(PyQt5.QtCore)
        except:
            return
    elif qt_support_mode == 'pyqt4':
        patch_qt_on_import = 'PyQt4'

        def get_qt_core_module():
            import PyQt4.QtCore
            return PyQt4.QtCore
        _patch_import_to_patch_pyqt_on_import(patch_qt_on_import, get_qt_core_module)
    else:
        raise ValueError('Unexpected qt support mode: %s' % (qt_support_mode,))