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
def get_build_args(compiler, optimization_level: str, debug: bool, sanitize: str) -> T.List[str]:
    build_args = compiler.get_optimization_args(optimization_level)
    build_args += compiler.get_debug_args(debug)
    build_args += compiler.sanitizer_compile_args(sanitize)
    return build_args