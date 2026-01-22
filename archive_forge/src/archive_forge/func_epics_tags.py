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
def epics_tags(self):
    """Return consolidated metadata from EPICS areaDetector tags as dict.

        Remove areaDetector tags from self.tags.

        """
    if not self.is_epics:
        return
    result = {}
    tags = self.tags
    for tag in list(self.tags.values()):
        code = tag.code
        if not 65000 <= code < 65500:
            continue
        value = tag.value
        if code == 65000:
            result['timeStamp'] = datetime.datetime.fromtimestamp(float(value))
        elif code == 65001:
            result['uniqueID'] = int(value)
        elif code == 65002:
            result['epicsTSSec'] = int(value)
        elif code == 65003:
            result['epicsTSNsec'] = int(value)
        else:
            key, value = value.split(':', 1)
            result[key] = astype(value)
        del tags[tag.name]
    return result