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
def stk_description_metadata(description):
    """Return metadata from MetaMorph image description as list of dict.

    The MetaMorph image description format is unspecified. Expect failures.

    """
    description = description.strip()
    if not description:
        return []
    try:
        description = bytes2str(description)
    except UnicodeDecodeError:
        warnings.warn('failed to parse MetaMorph image description')
        return []
    result = []
    for plane in description.split('\x00'):
        d = {}
        for line in plane.split('\r\n'):
            line = line.split(':', 1)
            if len(line) > 1:
                name, value = line
                d[name.strip()] = astype(value.strip())
            else:
                value = line[0].strip()
                if value:
                    if '' in d:
                        d[''].append(value)
                    else:
                        d[''] = [value]
        result.append(d)
    return result