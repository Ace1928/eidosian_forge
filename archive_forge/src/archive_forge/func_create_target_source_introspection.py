from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, unique
from functools import lru_cache
from pathlib import PurePath, Path
from textwrap import dedent
import itertools
import json
import os
import pickle
import re
import subprocess
import typing as T
from . import backends
from .. import modules
from .. import environment, mesonlib
from .. import build
from .. import mlog
from .. import compilers
from ..arglist import CompilerArgs
from ..compilers import Compiler
from ..linkers import ArLikeLinker, RSPFileSyntax
from ..mesonlib import (
from ..mesonlib import get_compiler_for_source, has_path_sep, OptionKey
from .backends import CleanTrees
from ..build import GeneratedList, InvalidArguments
def create_target_source_introspection(self, target: build.Target, comp: compilers.Compiler, parameters, sources, generated_sources, unity_sources: T.Optional[T.List[mesonlib.FileOrString]]=None):
    """
        Adds the source file introspection information for a language of a target

        Internal introspection storage format:
        self.introspection_data = {
            '<target ID>': {
                <id tuple>: {
                    'language: 'lang',
                    'compiler': ['comp', 'exe', 'list'],
                    'parameters': ['UNIQUE', 'parameter', 'list'],
                    'sources': [],
                    'generated_sources': [],
                }
            }
        }
        """
    tid = target.get_id()
    lang = comp.get_language()
    tgt = self.introspection_data[tid]
    id_hash = (lang, tuple(parameters))
    src_block = tgt.get(id_hash, None)
    if src_block is None:
        if isinstance(parameters, CompilerArgs):
            parameters = parameters.to_native(copy=True)
        parameters = comp.compute_parameters_with_absolute_paths(parameters, self.build_dir)
        src_block = {'language': lang, 'compiler': comp.get_exelist(), 'parameters': parameters, 'sources': [], 'generated_sources': [], 'unity_sources': []}
        tgt[id_hash] = src_block

    def compute_path(file: mesonlib.FileOrString) -> str:
        """ Make source files absolute """
        if isinstance(file, File):
            return file.absolute_path(self.source_dir, self.build_dir)
        return os.path.normpath(os.path.join(self.build_dir, file))
    src_block['sources'].extend((compute_path(x) for x in sources))
    src_block['generated_sources'].extend((compute_path(x) for x in generated_sources))
    if unity_sources:
        src_block['unity_sources'].extend((compute_path(x) for x in unity_sources))