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
def offsets_bytecounts(self):
    """Return simplified offsets and bytecounts."""
    if self.keyframe.is_contiguous:
        return (self.dataoffsets[:1], self.keyframe.is_contiguous[1:])
    return clean_offsets_counts(self.dataoffsets, self.databytecounts)