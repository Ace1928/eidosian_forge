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
def list_deps_from_source(intr: IntrospectionInterpreter) -> T.List[T.Dict[str, T.Union[str, bool]]]:
    result: T.List[T.Dict[str, T.Union[str, bool]]] = []
    for i in intr.dependencies:
        keys = ['name', 'required', 'version', 'has_fallback', 'conditional']
        result += [{k: v for k, v in i.items() if k in keys}]
    return result