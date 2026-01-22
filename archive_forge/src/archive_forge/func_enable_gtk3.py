import sys
import select
def enable_gtk3(self, app=None):
    """Enable event loop integration with Gtk3 (gir bindings).

        Parameters
        ----------
        app : ignored
           Ignored, it's only a placeholder to keep the call signature of all
           gui activation methods consistent, which simplifies the logic of
           supporting magics.

        Notes
        -----
        This methods sets the PyOS_InputHook for Gtk3, which allows
        the Gtk3 to integrate with terminal based applications like
        IPython.
        """
    from pydev_ipython.inputhookgtk3 import create_inputhook_gtk3
    self.set_inputhook(create_inputhook_gtk3(self._stdin_file))
    self._current_gui = GUI_GTK