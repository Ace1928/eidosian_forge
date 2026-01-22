from __future__ import annotations, division
import ast
import functools
import hashlib
import inspect
import os
import textwrap
from collections import defaultdict, namedtuple
from functools import cached_property
from typing import Callable, Generic, Iterable, List, Optional, TypeVar, Union, cast, overload
from .._C.libtriton.triton import TMAInfos
from ..common.backend import get_backend, get_cuda_version_key
from .interpreter import InterpretedFunction
@property
def cache_key(self):
    if self.hash is None:
        dependencies_finder = DependenciesFinder(globals=self.__globals__, src=self.src)
        dependencies_finder.visit(self.parse())
        self.hash = dependencies_finder.ret + str(self.starting_line_number)
    return self.hash