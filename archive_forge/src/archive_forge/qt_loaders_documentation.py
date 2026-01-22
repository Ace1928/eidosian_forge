import sys
from functools import partial
from pydev_ipython.version import check_version

    Attempt to import Qt, given a preference list
    of permissible bindings

    It is safe to call this function multiple times.

    Parameters
    ----------
    api_options: List of strings
        The order of APIs to try. Valid items are 'pyside',
        'pyqt', and 'pyqtv1'

    Returns
    -------

    A tuple of QtCore, QtGui, QtSvg, QT_API
    The first three are the Qt modules. The last is the
    string indicating which module was loaded.

    Raises
    ------
    ImportError, if it isn't possible to import any requested
    bindings (either becaues they aren't installed, or because
    an incompatible library has already been installed)
    