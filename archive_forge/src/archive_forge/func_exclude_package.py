import io
import itertools
import numbers
import os
import re
import sys
from contextlib import suppress
from glob import iglob
from pathlib import Path
from typing import List, Optional, Set
import distutils.cmd
import distutils.command
import distutils.core
import distutils.dist
import distutils.log
from distutils.debug import DEBUG
from distutils.errors import DistutilsOptionError, DistutilsSetupError
from distutils.fancy_getopt import translate_longopt
from distutils.util import strtobool
from .extern.more_itertools import partition, unique_everseen
from .extern.ordered_set import OrderedSet
from .extern.packaging.markers import InvalidMarker, Marker
from .extern.packaging.specifiers import InvalidSpecifier, SpecifierSet
from .extern.packaging.version import Version
from . import _entry_points
from . import _normalization
from . import _reqs
from . import command as _  # noqa  -- imported for side-effects
from ._importlib import metadata
from .config import setupcfg, pyprojecttoml
from .discovery import ConfigDiscovery
from .monkey import get_unpatched
from .warnings import InformationOnly, SetuptoolsDeprecationWarning
def exclude_package(self, package):
    """Remove packages, modules, and extensions in named package"""
    pfx = package + '.'
    if self.packages:
        self.packages = [p for p in self.packages if p != package and (not p.startswith(pfx))]
    if self.py_modules:
        self.py_modules = [p for p in self.py_modules if p != package and (not p.startswith(pfx))]
    if self.ext_modules:
        self.ext_modules = [p for p in self.ext_modules if p.name != package and (not p.name.startswith(pfx))]