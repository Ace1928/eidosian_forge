from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
@property
def is_eer(self) -> bool:
    """Page contains EER acquisition metadata."""
    return self.parent.is_bigtiff and self.compression in {1, 65000, 65001, 65002} and (65001 in self.tags) and (self.tags[65001].dtype == 7)