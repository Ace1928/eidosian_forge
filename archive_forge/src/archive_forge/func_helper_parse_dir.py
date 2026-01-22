from __future__ import annotations
from .common import CMakeException, CMakeBuildFile, CMakeConfiguration
import typing as T
from .. import mlog
from pathlib import Path
import json
import re
def helper_parse_dir(dir_entry: T.Dict[str, T.Any]) -> T.Tuple[Path, Path]:
    src_dir = Path(dir_entry.get('source', '.'))
    bld_dir = Path(dir_entry.get('build', '.'))
    src_dir = src_dir if src_dir.is_absolute() else source_dir / src_dir
    bld_dir = bld_dir if bld_dir.is_absolute() else build_dir / bld_dir
    src_dir = src_dir.resolve()
    bld_dir = bld_dir.resolve()
    return (src_dir, bld_dir)