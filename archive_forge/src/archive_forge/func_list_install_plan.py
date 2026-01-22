from __future__ import annotations
from contextlib import redirect_stdout
import collections
import dataclasses
import json
import os
from pathlib import Path, PurePath
import sys
import typing as T
from . import build, mesonlib, coredata as cdata
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstJSONPrinter
from .backend import backends
from .dependencies import Dependency
from . import environment
from .interpreterbase import ObjectHolder
from .mesonlib import OptionKey
from .mparser import FunctionNode, ArrayNode, ArgumentNode, BaseStringNode
def list_install_plan(installdata: backends.InstallData) -> T.Dict[str, T.Dict[str, T.Dict[str, T.Optional[str]]]]:
    plan: T.Dict[str, T.Dict[str, T.Dict[str, T.Optional[str]]]] = {'targets': {os.path.join(installdata.build_dir, target.fname): {'destination': target.out_name, 'tag': target.tag or None, 'subproject': target.subproject or None} for target in installdata.targets}}
    for key, data_list in {'data': installdata.data, 'man': installdata.man, 'headers': installdata.headers, 'install_subdirs': installdata.install_subdirs}.items():
        for data in data_list:
            data_type = data.data_type or key
            install_path_name = data.install_path_name
            if key == 'headers':
                install_path_name = os.path.join(install_path_name, os.path.basename(data.path))
            entry = {'destination': install_path_name, 'tag': data.tag or None, 'subproject': data.subproject or None}
            if key == 'install_subdirs':
                exclude_files, exclude_dirs = data.exclude or ([], [])
                entry['exclude_dirs'] = list(exclude_dirs)
                entry['exclude_files'] = list(exclude_files)
            plan[data_type] = plan.get(data_type, {})
            plan[data_type][data.path] = entry
    return plan