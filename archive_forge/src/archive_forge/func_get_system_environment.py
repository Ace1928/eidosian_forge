import os
import sys
import hashlib
import filecmp
from collections import namedtuple
from shutil import which
from jedi.cache import memoize_method, time_cache
from jedi.inference.compiled.subprocess import CompiledSubprocess, \
import parso
def get_system_environment(version, *, env_vars=None):
    """
    Return the first Python environment found for a string of the form 'X.Y'
    where X and Y are the major and minor versions of Python.

    :raises: :exc:`.InvalidPythonEnvironment`
    :returns: :class:`.Environment`
    """
    exe = which('python' + version)
    if exe:
        if exe == sys.executable:
            return SameEnvironment()
        return Environment(exe)
    if os.name == 'nt':
        for exe in _get_executables_from_windows_registry(version):
            try:
                return Environment(exe, env_vars=env_vars)
            except InvalidPythonEnvironment:
                pass
    raise InvalidPythonEnvironment('Cannot find executable python%s.' % version)