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
def svs_description_metadata(description):
    """Return metatata from Aperio image description as dict.

    The Aperio image description format is unspecified. Expect failures.

    >>> svs_description_metadata('Aperio Image Library v1.0')
    {'Aperio Image Library': 'v1.0'}

    """
    if not description.startswith('Aperio Image Library '):
        raise ValueError('invalid Aperio image description')
    result = {}
    lines = description.split('\n')
    key, value = lines[0].strip().rsplit(None, 1)
    result[key.strip()] = value.strip()
    if len(lines) == 1:
        return result
    items = lines[1].split('|')
    result[''] = items[0].strip()
    for item in items[1:]:
        key, value = item.split(' = ')
        result[key.strip()] = astype(value.strip())
    return result