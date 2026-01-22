from jnius import autoclass, PythonJavaClass, java_method
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Fbo, Callback, Rectangle
from kivy.core.camera import CameraBase
import threading
def read_frame(self):
    """
        Grab and decode frame in one call
        """
    return self.decode_frame(self.grab_frame())