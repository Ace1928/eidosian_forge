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
def get_cmdline_options(self):
    """Return a '{cmd: {opt:val}}' map of all command-line options

        Option names are all long, but do not include the leading '--', and
        contain dashes rather than underscores.  If the option doesn't take
        an argument (e.g. '--quiet'), the 'val' is 'None'.

        Note that options provided by config files are intentionally excluded.
        """
    d = {}
    for cmd, opts in self.command_options.items():
        for opt, (src, val) in opts.items():
            if src != 'command line':
                continue
            opt = opt.replace('_', '-')
            if val == 0:
                cmdobj = self.get_command_obj(cmd)
                neg_opt = self.negative_opt.copy()
                neg_opt.update(getattr(cmdobj, 'negative_opt', {}))
                for neg, pos in neg_opt.items():
                    if pos == opt:
                        opt = neg
                        val = None
                        break
                else:
                    raise AssertionError("Shouldn't be able to get here")
            elif val == 1:
                val = None
            d.setdefault(cmd, {})[opt] = val
    return d