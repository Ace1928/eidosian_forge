import struct
from io import BytesIO
from functools import wraps
import warnings
def pack_farray(self, n, list, pack_item):
    if len(list) != n:
        raise ValueError('wrong array size')
    for item in list:
        pack_item(item)