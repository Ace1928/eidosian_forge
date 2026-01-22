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
def generate_solution_dirs(self, ofile: str, parents: T.Sequence[Path]) -> None:
    prj_templ = 'Project("{%s}") = "%s", "%s", "{%s}"\n'
    iterpaths = reversed(parents)
    next(iterpaths)
    for path in iterpaths:
        if path not in self.subdirs:
            basename = path.name
            identifier = generate_guid_from_path(path, 'subdir')
            parent_dir = path.parent
            parent_identifier = self.subdirs[parent_dir][0] if parent_dir != PurePath('.') else None
            self.subdirs[path] = (identifier, parent_identifier)
            prj_line = prj_templ % (self.environment.coredata.lang_guids['directory'], basename, basename, self.subdirs[path][0])
            ofile.write(prj_line)
            ofile.write('EndProject\n')