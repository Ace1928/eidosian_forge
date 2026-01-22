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
def nih_metadata(self):
    """Return NIH Image metadata from NIHImageHeader tag as dict."""
    if not self.is_nih:
        return
    return self.pages[0].tags['NIHImageHeader'].value