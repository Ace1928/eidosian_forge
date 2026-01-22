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
def overwrite_description(self, description: str, /) -> None:
    """Overwrite value of last ImageDescription tag.

        Can be used to write OME-XML after writing images.
        Ends a contiguous series.

        """
    if self._descriptiontag is None:
        raise ValueError('no ImageDescription tag found')
    self._write_remaining_pages()
    self._descriptiontag.overwrite(description, erase=False)
    self._descriptiontag = None