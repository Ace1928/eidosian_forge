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
def relocate_generated_file_paths_to_concrete_build_dir(self, gen_files: T.List[str], target: T.Union[build.Target, build.CustomTargetIndex]) -> None:
    _, build_dir_tail = os.path.split(self.src_to_build)
    meson_build_dir_for_buildtype = build_dir_tail[:-2] + coredata.get_genvs_default_buildtype_list()[0]
    proj_to_build_root = self.target_to_build_root(target)
    proj_to_multiconfigured_builds_parent_dir = os.path.join(proj_to_build_root, '..')
    proj_to_build_dir_for_buildtype = str(os.path.join(proj_to_multiconfigured_builds_parent_dir, meson_build_dir_for_buildtype))
    relocate_to_concrete_builddir_target = os.path.normpath(os.path.join(proj_to_build_dir_for_buildtype, self.get_target_dir(target)))
    for idx, file_path in enumerate(gen_files):
        gen_files[idx] = os.path.normpath(os.path.join(relocate_to_concrete_builddir_target, file_path))