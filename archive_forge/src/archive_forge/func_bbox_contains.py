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
def bbox_contains(bbox1, bbox2):
    """Tests whether bbox1 fully contains bbox2, returning a boolean
    """
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    contains = xmin1 < xmin2 and xmax1 > xmax2 and (ymin1 < ymin2) and (ymax1 > ymax2)
    return contains