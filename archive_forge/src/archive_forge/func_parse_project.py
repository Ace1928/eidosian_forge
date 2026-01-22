from __future__ import annotations
from .common import CMakeException, CMakeBuildFile, CMakeConfiguration
import typing as T
from .. import mlog
from pathlib import Path
import json
import re
def parse_project(pro: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
    p_src_dir = source_dir
    p_bld_dir = build_dir
    try:
        p_src_dir, p_bld_dir = helper_parse_dir(cnf['directories'][pro['directoryIndexes'][0]])
    except (IndexError, KeyError):
        pass
    pro_data = {'name': pro.get('name', ''), 'sourceDirectory': p_src_dir, 'buildDirectory': p_bld_dir, 'targets': []}
    for ref in pro.get('targetIndexes', []):
        tgt = {}
        try:
            tgt = cnf['targets'][ref]
        except (IndexError, KeyError):
            pass
        pro_data['targets'] += [parse_target(tgt)]
    return pro_data