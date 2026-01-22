import array
import os
import struct
import six
from ._exceptions import *
from ._utils import validate_utf8
from threading import Lock
def is_fire(self, frame):
    return frame.fin or self.fire_cont_frame