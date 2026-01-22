from jnius import autoclass, PythonJavaClass, java_method
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Fbo, Callback, Rectangle
from kivy.core.camera import CameraBase
import threading
def decode_frame(self, buf):
    """
        Decode image data from grabbed frame.

        This method depends on OpenCV and NumPy - however it is only used for
        fetching the current frame as a NumPy array, and not required when
        this :class:`CameraAndroid` provider is simply used by a
        :class:`~kivy.uix.camera.Camera` widget.
        """
    import numpy as np
    from cv2 import cvtColor
    w, h = self._resolution
    arr = np.fromstring(buf, 'uint8').reshape((h + h / 2, w))
    arr = cvtColor(arr, 93)
    return arr