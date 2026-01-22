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
def find_buildsystem_files_list(src_dir: str) -> T.List[str]:
    build_files = frozenset({'meson.build', 'meson.options', 'meson_options.txt'})
    filelist: T.List[str] = []
    for root, _, files in os.walk(src_dir):
        filelist.extend((os.path.relpath(os.path.join(root, f), src_dir) for f in build_files.intersection(files)))
    return filelist