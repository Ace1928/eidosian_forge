import sys
import select
def disable_pyglet(self):
    """Disable event loop integration with pyglet.

        This merely sets PyOS_InputHook to NULL.
        """
    self.clear_inputhook()