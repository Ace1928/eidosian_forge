import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
@staticmethod
def image_profile(stream, keep_image=0):
    """
        Metadata of an image binary stream.
        """
    return JM_image_profile(stream, keep_image)