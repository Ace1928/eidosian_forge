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
def get_obj_target_deps(self, obj_list):
    result = {}
    for o in obj_list:
        if isinstance(o, build.ExtractedObjects):
            result[o.target.get_id()] = o.target
    return result.items()