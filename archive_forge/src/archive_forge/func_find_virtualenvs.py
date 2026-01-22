import os
import sys
import hashlib
import filecmp
from collections import namedtuple
from shutil import which
from jedi.cache import memoize_method, time_cache
from jedi.inference.compiled.subprocess import CompiledSubprocess, \
import parso
def find_virtualenvs(paths=None, *, safe=True, use_environment_vars=True):
    """
    :param paths: A list of paths in your file system to be scanned for
        Virtualenvs. It will search in these paths and potentially execute the
        Python binaries.
    :param safe: Default True. In case this is False, it will allow this
        function to execute potential `python` environments. An attacker might
        be able to drop an executable in a path this function is searching by
        default. If the executable has not been installed by root, it will not
        be executed.
    :param use_environment_vars: Default True. If True, the VIRTUAL_ENV
        variable will be checked if it contains a valid VirtualEnv.
        CONDA_PREFIX will be checked to see if it contains a valid conda
        environment.

    :yields: :class:`.Environment`
    """
    if paths is None:
        paths = []
    _used_paths = set()
    if use_environment_vars:
        virtual_env = _get_virtual_env_from_var()
        if virtual_env is not None:
            yield virtual_env
            _used_paths.add(virtual_env.path)
        conda_env = _get_virtual_env_from_var(_CONDA_VAR)
        if conda_env is not None:
            yield conda_env
            _used_paths.add(conda_env.path)
    for directory in paths:
        if not os.path.isdir(directory):
            continue
        directory = os.path.abspath(directory)
        for path in os.listdir(directory):
            path = os.path.join(directory, path)
            if path in _used_paths:
                continue
            _used_paths.add(path)
            try:
                executable = _get_executable_path(path, safe=safe)
                yield Environment(executable)
            except InvalidPythonEnvironment:
                pass