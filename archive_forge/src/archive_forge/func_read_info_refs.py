import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
def read_info_refs(f):
    ret = {}
    for line in f.readlines():
        sha, name = line.rstrip(b'\r\n').split(b'\t', 1)
        ret[name] = sha
    return ret