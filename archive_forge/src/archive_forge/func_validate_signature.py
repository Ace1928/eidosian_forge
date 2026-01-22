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
def validate_signature(self):
    """
        If signature (header) has not been read then read and
        validate it; otherwise do nothing.
        """
    if self.signature:
        return
    self.signature = self.file.read(8)
    if self.signature != signature:
        raise FormatError('PNG file has invalid signature.')