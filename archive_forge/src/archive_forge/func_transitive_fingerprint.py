from __future__ import absolute_import, print_function
import cython
from .. import __version__
import collections
import contextlib
import hashlib
import os
import shutil
import subprocess
import re, sys, time
from glob import iglob
from io import open as io_open
from os.path import relpath as _relpath
import zipfile
from .. import Utils
from ..Utils import (cached_function, cached_method, path_exists,
from ..Compiler import Errors
from ..Compiler.Main import Context
from ..Compiler.Options import (CompilationOptions, default_options,
def transitive_fingerprint(self, filename, module, compilation_options):
    """
        Return a fingerprint of a cython file that is about to be cythonized.

        Fingerprints are looked up in future compilations. If the fingerprint
        is found, the cythonization can be skipped. The fingerprint must
        incorporate everything that has an influence on the generated code.
        """
    try:
        m = hashlib.sha1(__version__.encode('UTF-8'))
        m.update(file_hash(filename).encode('UTF-8'))
        for x in sorted(self.all_dependencies(filename)):
            if os.path.splitext(x)[1] not in ('.c', '.cpp', '.h'):
                m.update(file_hash(x).encode('UTF-8'))
        m.update(str((module.language, getattr(module, 'py_limited_api', False), getattr(module, 'np_pythran', False))).encode('UTF-8'))
        m.update(compilation_options.get_fingerprint().encode('UTF-8'))
        return m.hexdigest()
    except IOError:
        return None