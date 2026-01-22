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
def shapeRecords(self, fields=None, bbox=None):
    """Returns a list of combination geometry/attribute records for
        all records in a shapefile.
        To only read some of the fields, specify the 'fields' arg as a
        list of one or more fieldnames. 
        To only read entries within a given spatial region, specify the 'bbox'
        arg as a list or tuple of xmin,ymin,xmax,ymax. 
        """
    return ShapeRecords(self.iterShapeRecords(fields=fields, bbox=bbox))