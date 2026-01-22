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
def shapeRecord(self, i=0, fields=None, bbox=None):
    """Returns a combination geometry and attribute record for the
        supplied record index. 
        To only read some of the fields, specify the 'fields' arg as a
        list of one or more fieldnames. 
        If the 'bbox' arg is given (list or tuple of xmin,ymin,xmax,ymax), 
        returns None if the shape is not within that region. 
        """
    i = self.__restrictIndex(i)
    shape = self.shape(i, bbox=bbox)
    if shape:
        record = self.record(i, fields=fields)
        return ShapeRecord(shape=shape, record=record)