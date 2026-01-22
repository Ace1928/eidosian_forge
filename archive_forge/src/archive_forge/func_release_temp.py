from __future__ import absolute_import
import cython
import hashlib
import operator
import os
import re
import shutil
import textwrap
from string import Template
from functools import partial
from contextlib import closing, contextmanager
from collections import defaultdict
from . import Naming
from . import Options
from . import DebugFlags
from . import StringEncoding
from .. import Utils
from .Scanning import SourceDescriptor
from ..StringIOTree import StringIOTree
def release_temp(self, name):
    """
        Releases a temporary so that it can be reused by other code needing
        a temp of the same type.
        """
    type, manage_ref = self.temps_used_type[name]
    freelist = self.temps_free.get((type, manage_ref))
    if freelist is None:
        freelist = ([], set())
        self.temps_free[type, manage_ref] = freelist
    if name in freelist[1]:
        raise RuntimeError('Temp %s freed twice!' % name)
    if name not in self.zombie_temps:
        freelist[0].append(name)
    freelist[1].add(name)
    if DebugFlags.debug_temp_code_comments:
        self.owner.putln('/* %s released %s*/' % (name, ' - zombie' if name in self.zombie_temps else ''))