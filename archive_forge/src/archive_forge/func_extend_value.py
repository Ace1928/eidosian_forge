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
def extend_value(self, value: T.Union[str, T.List[str]]) -> None:
    """Extend the value with an additional value."""
    new = self.validate_value(value)
    self.set_value(self.value + new)