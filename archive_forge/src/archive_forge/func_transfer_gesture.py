import pickle
import base64
import zlib
from re import match as re_match
from collections import deque
from math import sqrt, pi, radians, acos, atan, atan2, pow, floor
from math import sin as math_sin, cos as math_cos
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.properties import ListProperty
from kivy.compat import PY2
from io import BytesIO
def transfer_gesture(self, tgt, **kwargs):
    """Transfers :class:`MultistrokeGesture` objects from
        :attr:`Recognizer.db` to another :class:`Recognizer` instance `tgt`.

        This method accepts optional :meth:`Recognizer.filter` arguments.
        """
    if hasattr(tgt, 'db') and isinstance(tgt.db, list):
        send = self.filter(**kwargs)
        if send:
            tgt.db.append(None)
            tgt.db[-1:] = send
            return True