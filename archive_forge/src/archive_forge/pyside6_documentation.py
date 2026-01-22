from PySide6 import QtCore  # pylint: disable=import-error
from ._qt_base import MonitorObserverGenerator

    pyudev.pyside
    =============

    PySide integration.

    :class:`QUDevMonitorObserver` integrates device monitoring into the
    PySide\_ mainloop by turing device events into Qt signals.

    :mod:`PySide.QtCore` from PySide\_ must be available when importing this
    module.

    .. _PySide: http://www.pyside.org

    .. moduleauthor::  Sebastian Wiesner  <lunaryorn@gmail.com>
    .. versionadded:: 0.6
