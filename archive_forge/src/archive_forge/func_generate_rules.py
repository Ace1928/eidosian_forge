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
def generate_rules(self):
    self.rules = []
    self.ruledict = {}
    self.add_rule_comment(NinjaComment('Rules for module scanning.'))
    self.generate_scanner_rules()
    self.add_rule_comment(NinjaComment('Rules for compiling.'))
    self.generate_compile_rules()
    self.add_rule_comment(NinjaComment('Rules for linking.'))
    self.generate_static_link_rules()
    self.generate_dynamic_link_rules()
    self.add_rule_comment(NinjaComment('Other rules'))
    self.add_rule(NinjaRule('CUSTOM_COMMAND', ['$COMMAND'], [], '$DESC', extra='restat = 1'))
    self.add_rule(NinjaRule('CUSTOM_COMMAND_DEP', ['$COMMAND'], [], '$DESC', deps='gcc', depfile='$DEPFILE', extra='restat = 1'))
    self.add_rule(NinjaRule('COPY_FILE', self.environment.get_build_command() + ['--internal', 'copy'], ['$in', '$out'], 'Copying $in to $out'))
    c = self.environment.get_build_command() + ['--internal', 'regenerate', self.environment.get_source_dir(), '.']
    self.add_rule(NinjaRule('REGENERATE_BUILD', c, [], 'Regenerating build files.', extra='generator = 1'))