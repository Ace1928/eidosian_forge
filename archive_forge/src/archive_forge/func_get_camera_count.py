from jnius import autoclass, PythonJavaClass, java_method
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Fbo, Callback, Rectangle
from kivy.core.camera import CameraBase
import threading
@staticmethod
def get_camera_count():
    """
        Get the number of available cameras.
        """
    return Camera.getNumberOfCameras()