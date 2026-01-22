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
def export_gesture(self, filename=None, **kwargs):
    """Export a list of :class:`MultistrokeGesture` objects. Outputs a
        base64-encoded string that can be decoded to a Python list with
        the :meth:`parse_gesture` function or imported directly to
        :attr:`self.db` using :meth:`Recognizer.import_gesture`. If
        `filename` is specified, the output is written to disk, otherwise
        returned.

        This method accepts optional :meth:`Recognizer.filter` arguments.
        """
    io = BytesIO()
    p = pickle.Pickler(io, protocol=0)
    multistrokes = []
    defaults = {'priority': 100, 'numpoints': 16, 'stroke_sens': True, 'orientation_sens': False, 'angle_similarity': 30.0}
    dkeys = defaults.keys()
    for multistroke in self.filter(**kwargs):
        m = dict(defaults)
        m = {'name': multistroke.name}
        for attr in dkeys:
            m[attr] = getattr(multistroke, attr)
        m['strokes'] = tuple(([(p.x, p.y) for p in line] for line in multistroke.strokes))
        multistrokes.append(m)
    p.dump(multistrokes)
    if filename:
        f = open(filename, 'wb')
        f.write(base64.b64encode(zlib.compress(io.getvalue(), 9)))
        f.close()
    else:
        return base64.b64encode(zlib.compress(io.getvalue(), 9))