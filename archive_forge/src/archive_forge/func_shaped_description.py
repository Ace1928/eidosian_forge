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
@cached_property
def shaped_description(self) -> str | None:
    """Description containing array shape if exists, else None."""
    for description in (self.description, self.description1):
        if not description or '"mibi.' in description:
            return None
        if description[:1] == '{' and '"shape":' in description:
            return description
        if description[:6] == 'shape=':
            return description
    return None