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
@property
def shapeTypeName(self):
    return SHAPETYPE_LOOKUP[self.shapeType]