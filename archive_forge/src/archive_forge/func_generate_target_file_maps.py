from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def generate_target_file_maps(self) -> None:
    self.generate_target_file_maps_impl(self.build_targets)
    self.generate_target_file_maps_impl(self.custom_targets)