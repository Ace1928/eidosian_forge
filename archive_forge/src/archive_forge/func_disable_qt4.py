import sys
import select
def disable_qt4(self):
    """Disable event loop integration with PyQt4.

        This merely sets PyOS_InputHook to NULL.
        """
    if GUI_QT4 in self._apps:
        self._apps[GUI_QT4]._in_event_loop = False
    self.clear_inputhook()