from __future__ import annotations
import os
import re
import subprocess
import typing as T
from .. import mlog
from .. import mesonlib
from ..compilers.compilers import CrossNoRunException
from ..mesonlib import (
from ..environment import detect_cpu_family
from .base import DependencyException, DependencyMethods, DependencyTypeName, SystemDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory
@staticmethod
def weird_filter(elems: T.List[str]) -> T.List[str]:
    """When building packages, the output of the enclosing Make is
        sometimes mixed among the subprocess output. I have no idea why. As a
        hack filter out everything that is not a flag.
        """
    return [e for e in elems if e.startswith('-')]