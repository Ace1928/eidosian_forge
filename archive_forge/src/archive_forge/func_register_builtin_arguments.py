from __future__ import annotations
import copy
from . import mlog, mparser
import pickle, os, uuid
import sys
from itertools import chain
from pathlib import PurePath
from collections import OrderedDict, abc
from dataclasses import dataclass
from .mesonlib import (
from .wrap import WrapMode
import ast
import argparse
import configparser
import enum
import shlex
import typing as T
def register_builtin_arguments(parser: argparse.ArgumentParser) -> None:
    for n, b in BUILTIN_OPTIONS.items():
        b.add_to_argparse(str(n), parser, '')
    for n, b in BUILTIN_OPTIONS_PER_MACHINE.items():
        b.add_to_argparse(str(n), parser, ' (just for host machine)')
        b.add_to_argparse(str(n.as_build()), parser, ' (just for build machine)')
    parser.add_argument('-D', action='append', dest='projectoptions', default=[], metavar='option', help='Set the value of an option, can be used several times to set multiple options.')