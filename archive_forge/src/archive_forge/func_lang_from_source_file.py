from __future__ import annotations
import copy
import itertools
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
import uuid
import typing as T
from pathlib import Path, PurePath, PureWindowsPath
import re
from collections import Counter
from . import backends
from .. import build
from .. import mlog
from .. import compilers
from .. import mesonlib
from ..mesonlib import (
from ..environment import Environment, build_filename
from .. import coredata
@classmethod
def lang_from_source_file(cls, src):
    ext = src.split('.')[-1]
    if ext in compilers.c_suffixes:
        return 'c'
    if ext in compilers.cpp_suffixes:
        return 'cpp'
    raise MesonException(f'Could not guess language from source file {src}.')