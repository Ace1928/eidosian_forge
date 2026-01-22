import pickle
import base64
import zlib
import math
from kivy.vector import Vector
from io import BytesIO
def center_stroke(self, offset_x, offset_y):
    """Centers the stroke by offsetting the points."""
    for point in self.points:
        point.x -= offset_x
        point.y -= offset_y