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
def function2(position):

    class Position2:
        pass
    position2 = Position2()
    position2.depth = position.depth
    position2.heading = position.heading
    position2.id = position.id
    position2.rect = JM_py_from_rect(position.rect)
    position2.text = position.text
    position2.open_close = position.open_close
    position2.rect_num = position.rectangle_num
    position2.href = position.href
    if args:
        for k, v in args.items():
            setattr(position2, k, v)
    function(position2)