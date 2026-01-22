import pickle
import base64
import zlib
import math
from kivy.vector import Vector
from io import BytesIO
def scale_stroke(self, scale_factor):
    """
        scale_stroke(scale_factor=float)
        Scales the stroke down by scale_factor.
        """
    self.points = [pt.scale(scale_factor) for pt in self.points]