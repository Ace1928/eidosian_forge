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
def load_shp(self, shapefile_name):
    """
        Attempts to load file with .shp extension as both lower and upper case
        """
    shp_ext = 'shp'
    try:
        self.shp = open('%s.%s' % (shapefile_name, shp_ext), 'rb')
        self._files_to_close.append(self.shp)
    except IOError:
        try:
            self.shp = open('%s.%s' % (shapefile_name, shp_ext.upper()), 'rb')
            self._files_to_close.append(self.shp)
        except IOError:
            pass