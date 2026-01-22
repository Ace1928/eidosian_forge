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
def iterShapeRecords(self, fields=None, bbox=None):
    """Returns a generator of combination geometry/attribute records for
        all records in a shapefile.
        To only read some of the fields, specify the 'fields' arg as a
        list of one or more fieldnames. 
        To only read entries within a given spatial region, specify the 'bbox'
        arg as a list or tuple of xmin,ymin,xmax,ymax. 
        """
    if bbox is None:
        for shape, record in izip(self.iterShapes(), self.iterRecords(fields=fields)):
            yield ShapeRecord(shape=shape, record=record)
    else:
        for shape in self.iterShapes(bbox=bbox):
            if shape:
                record = self.record(i=shape.oid, fields=fields)
                yield ShapeRecord(shape=shape, record=record)