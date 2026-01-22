from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
@lazyattr
def qptiff_metadata(self):
    """Return PerkinElmer-QPI-ImageDescription XML element as dict."""
    if not self.is_qptiff:
        return
    root = 'PerkinElmer-QPI-ImageDescription'
    xml = self.pages[0].description.replace(' ' + root + ' ', root)
    return xml2dict(xml)[root]