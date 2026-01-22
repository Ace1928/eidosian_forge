import importlib
import os
from typing import Tuple, Callable
def set_qt_api(gui):
    """Sets the `QT_API` environment variable if it isn't already set."""
    qt_api = os.environ.get('QT_API', None)
    from IPython.external.qt_loaders import QT_API_PYQT, QT_API_PYQT5, QT_API_PYQT6, QT_API_PYSIDE, QT_API_PYSIDE2, QT_API_PYSIDE6, QT_API_PYQTv1, loaded_api
    loaded = loaded_api()
    qt_env2gui = {QT_API_PYSIDE: 'qt4', QT_API_PYQTv1: 'qt4', QT_API_PYQT: 'qt4', QT_API_PYSIDE2: 'qt5', QT_API_PYQT5: 'qt5', QT_API_PYSIDE6: 'qt6', QT_API_PYQT6: 'qt6'}
    if loaded is not None and gui != 'qt':
        if qt_env2gui[loaded] != gui:
            print(f'Cannot switch Qt versions for this session; will use {qt_env2gui[loaded]}.')
            return qt_env2gui[loaded]
    if qt_api is not None and gui != 'qt':
        if qt_env2gui[qt_api] != gui:
            print(f'Request for "{gui}" will be ignored because `QT_API` environment variable is set to "{qt_api}"')
            return qt_env2gui[qt_api]
    else:
        if gui == 'qt5':
            try:
                import PyQt5
                os.environ['QT_API'] = 'pyqt5'
            except ImportError:
                try:
                    import PySide2
                    os.environ['QT_API'] = 'pyside2'
                except ImportError:
                    os.environ['QT_API'] = 'pyqt5'
        elif gui == 'qt6':
            try:
                import PyQt6
                os.environ['QT_API'] = 'pyqt6'
            except ImportError:
                try:
                    import PySide6
                    os.environ['QT_API'] = 'pyside6'
                except ImportError:
                    os.environ['QT_API'] = 'pyqt6'
        elif gui == 'qt':
            if 'QT_API' in os.environ.keys():
                del os.environ['QT_API']
        else:
            print(f'Unrecognized Qt version: {gui}. Should be "qt5", "qt6", or "qt".')
            return
        from IPython.external.qt_for_kernel import QT_API
        return qt_env2gui[QT_API]