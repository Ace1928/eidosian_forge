import sys
import os
from pathlib import Path
import io
def asunicode_nested(x):
    if hasattr(x, '__iter__') and (not isinstance(x, (bytes, unicode))):
        return [asunicode_nested(y) for y in x]
    else:
        return asunicode(x)