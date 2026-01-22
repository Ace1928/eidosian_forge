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
def scale_dim(points, size, oneDratio):
    bbox_x, bbox_y, bbox_w, bbox_h = bounding_box(points)
    if bbox_h == 0 or bbox_w == 0:
        raise MultistrokeError('scale_dim() called with invalid points: h:{}, w:{}'.format(bbox_h, bbox_w))
    uniformly = min(bbox_w / bbox_h, bbox_h / bbox_w) <= oneDratio
    if uniformly:
        qx_size = size / max(bbox_w, bbox_h)
        qy_size = size / max(bbox_w, bbox_h)
    else:
        qx_size = size / bbox_w
        qy_size = size / bbox_h
    newpoints = []
    newpoints_append = newpoints.append
    for p in points:
        qx = p[0] * qx_size
        qy = p[1] * qy_size
        newpoints_append(Vector(qx, qy))
    return newpoints