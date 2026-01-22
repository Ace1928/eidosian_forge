import sys
import select
def enable_pyglet(self, app=None):
    """Enable event loop integration with pyglet.

        Parameters
        ----------
        app : ignored
           Ignored, it's only a placeholder to keep the call signature of all
           gui activation methods consistent, which simplifies the logic of
           supporting magics.

        Notes
        -----
        This methods sets the ``PyOS_InputHook`` for pyglet, which allows
        pyglet to integrate with terminal based applications like
        IPython.

        """
    from pydev_ipython.inputhookpyglet import inputhook_pyglet
    self.set_inputhook(inputhook_pyglet)
    self._current_gui = GUI_PYGLET
    return app