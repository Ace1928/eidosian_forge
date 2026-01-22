import sys
import select
def disable_tk(self):
    """Disable event loop integration with Tkinter.

        This merely sets PyOS_InputHook to NULL.
        """
    self.clear_inputhook()