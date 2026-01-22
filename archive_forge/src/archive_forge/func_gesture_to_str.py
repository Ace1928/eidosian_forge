import pickle
import base64
import zlib
import math
from kivy.vector import Vector
from io import BytesIO
def gesture_to_str(self, gesture):
    """Convert a gesture into a unique string."""
    io = BytesIO()
    p = pickle.Pickler(io)
    p.dump(gesture)
    data = base64.b64encode(zlib.compress(io.getvalue(), 9))
    return data