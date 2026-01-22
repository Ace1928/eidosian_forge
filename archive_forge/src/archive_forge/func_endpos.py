import binascii
import functools
import logging
import re
import string
import struct
import numpy as np
from matplotlib.cbook import _format_approx
from . import _api
def endpos(self):
    """Position one past the end of the token"""
    return self.pos + len(self.raw)