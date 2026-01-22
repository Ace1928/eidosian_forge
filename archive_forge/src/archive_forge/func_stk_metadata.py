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
def stk_metadata(self):
    """Return STK metadata from UIC tags as dict."""
    if not self.is_stk:
        return
    page = self.pages[0]
    tags = page.tags
    result = {}
    result['NumberPlanes'] = tags['UIC2tag'].count
    if page.description:
        result['PlaneDescriptions'] = page.description.split('\x00')
    if 'UIC1tag' in tags:
        result.update(tags['UIC1tag'].value)
    if 'UIC3tag' in tags:
        result.update(tags['UIC3tag'].value)
    if 'UIC4tag' in tags:
        result.update(tags['UIC4tag'].value)
    uic2tag = tags['UIC2tag'].value
    result['ZDistance'] = uic2tag['ZDistance']
    result['TimeCreated'] = uic2tag['TimeCreated']
    result['TimeModified'] = uic2tag['TimeModified']
    try:
        result['DatetimeCreated'] = numpy.array([julian_datetime(*dt) for dt in zip(uic2tag['DateCreated'], uic2tag['TimeCreated'])], dtype='datetime64[ns]')
        result['DatetimeModified'] = numpy.array([julian_datetime(*dt) for dt in zip(uic2tag['DateModified'], uic2tag['TimeModified'])], dtype='datetime64[ns]')
    except ValueError as e:
        warnings.warn('stk_metadata: %s' % e)
    return result