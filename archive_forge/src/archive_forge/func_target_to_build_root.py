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
def target_to_build_root(self, target):
    if self.get_target_dir(target) == '':
        return ''
    directories = os.path.normpath(self.get_target_dir(target)).split(os.sep)
    return os.sep.join(['..'] * len(directories))