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
def micromanager_metadata(self):
    """Return consolidated MicroManager metadata as dict."""
    if not self.is_micromanager:
        return
    result = read_micromanager_metadata(self._fh)
    result.update(self.pages[0].tags['MicroManagerMetadata'].value)
    return result