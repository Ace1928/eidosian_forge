from __future__ import annotations
import functools, json, os, textwrap
from pathlib import Path
import typing as T
from .. import mesonlib, mlog
from .base import process_method_kw, DependencyException, DependencyMethods, DependencyTypeName, ExternalDependency, SystemDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory
from .framework import ExtraFrameworkDependency
from .pkgconfig import PkgConfigDependency
from ..environment import detect_cpu_family
from ..programs import ExternalProgram
def wrap_in_pythons_pc_dir(name: str, env: 'Environment', kwargs: T.Dict[str, T.Any], installation: 'BasicPythonExternalProgram') -> 'ExternalDependency':
    if not pkg_libdir:
        empty = ExternalDependency(DependencyTypeName('pkgconfig'), env, {})
        empty.name = 'python'
        return empty
    old_pkg_libdir = os.environ.pop('PKG_CONFIG_LIBDIR', None)
    old_pkg_path = os.environ.pop('PKG_CONFIG_PATH', None)
    os.environ['PKG_CONFIG_LIBDIR'] = pkg_libdir
    try:
        return PythonPkgConfigDependency(name, env, kwargs, installation, True)
    finally:

        def set_env(name: str, value: str) -> None:
            if value is not None:
                os.environ[name] = value
            elif name in os.environ:
                del os.environ[name]
        set_env('PKG_CONFIG_LIBDIR', old_pkg_libdir)
        set_env('PKG_CONFIG_PATH', old_pkg_path)