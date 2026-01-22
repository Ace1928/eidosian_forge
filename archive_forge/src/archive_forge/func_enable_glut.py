import sys
import select
def enable_glut(self, app=None):
    """ Enable event loop integration with GLUT.

        Parameters
        ----------

        app : ignored
            Ignored, it's only a placeholder to keep the call signature of all
            gui activation methods consistent, which simplifies the logic of
            supporting magics.

        Notes
        -----

        This methods sets the PyOS_InputHook for GLUT, which allows the GLUT to
        integrate with terminal based applications like IPython. Due to GLUT
        limitations, it is currently not possible to start the event loop
        without first creating a window. You should thus not create another
        window but use instead the created one. See 'gui-glut.py' in the
        docs/examples/lib directory.

        The default screen mode is set to:
        glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH
        """
    import OpenGL.GLUT as glut
    from pydev_ipython.inputhookglut import glut_display_mode, glut_close, glut_display, glut_idle, inputhook_glut
    if GUI_GLUT not in self._apps:
        argv = getattr(sys, 'argv', [])
        glut.glutInit(argv)
        glut.glutInitDisplayMode(glut_display_mode)
        if bool(glut.glutSetOption):
            glut.glutSetOption(glut.GLUT_ACTION_ON_WINDOW_CLOSE, glut.GLUT_ACTION_GLUTMAINLOOP_RETURNS)
        glut.glutCreateWindow(argv[0] if len(argv) > 0 else '')
        glut.glutReshapeWindow(1, 1)
        glut.glutHideWindow()
        glut.glutWMCloseFunc(glut_close)
        glut.glutDisplayFunc(glut_display)
        glut.glutIdleFunc(glut_idle)
    else:
        glut.glutWMCloseFunc(glut_close)
        glut.glutDisplayFunc(glut_display)
        glut.glutIdleFunc(glut_idle)
    self.set_inputhook(inputhook_glut)
    self._current_gui = GUI_GLUT
    self._apps[GUI_GLUT] = True