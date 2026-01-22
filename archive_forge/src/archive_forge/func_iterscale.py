import collections
import io   # For io.BytesIO
import itertools
import math
import operator
import re
import struct
import sys
import warnings
import zlib
from array import array
fromarray = from_array
def iterscale():
    for row in pixels:
        yield [int(round(x * factor)) for x in row]