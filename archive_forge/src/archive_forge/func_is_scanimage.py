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
@property
def is_scanimage(self):
    """Page contains ScanImage metadata."""
    return self.description[:12] == 'state.config' or self.software[:22] == 'SI.LINE_FORMAT_VERSION' or 'scanimage.SI.' in self.description[-256:]