from struct import pack, unpack, calcsize, error, Struct
import os
import sys
import time
import array
import tempfile
import logging
import io
from datetime import date
import zipfile
def multipointm(self, points):
    """Creates a MULTIPOINTM shape.
        Points is a list of xym values.
        If the m (measure) value is not included, it defaults to None (NoData)."""
    shapeType = MULTIPOINTM
    points = [points]
    self._shapeparts(parts=points, shapeType=shapeType)