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
@staticmethod
def escape_preprocessor_define(define: str) -> str:
    table = str.maketrans({'%': '%25', '$': '%24', '@': '%40', "'": '%27', ';': '%3B', '?': '%3F', '*': '%2A', '\\': '\\\\'})
    return define.translate(table)