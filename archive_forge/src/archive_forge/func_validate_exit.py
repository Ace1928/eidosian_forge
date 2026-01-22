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
def validate_exit(self):
    if self.temps_allocated:
        leftovers = self.temps_in_use()
        if leftovers:
            msg = "TEMPGUARD: Temps left over at end of '%s': %s" % (self.scope.name, ', '.join(['%s [%s]' % (name, ctype) for name, ctype, is_pytemp in sorted(leftovers)]))
            raise RuntimeError(msg)