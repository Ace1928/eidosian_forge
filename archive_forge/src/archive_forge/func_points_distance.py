import pickle
import base64
import zlib
import math
from kivy.vector import Vector
from io import BytesIO
def points_distance(self, point1, point2):
    """
        points_distance(point1=GesturePoint, point2=GesturePoint)
        Returns the distance between two GesturePoints.
        """
    x = point1.x - point2.x
    y = point1.y - point2.y
    return math.sqrt(x * x + y * y)