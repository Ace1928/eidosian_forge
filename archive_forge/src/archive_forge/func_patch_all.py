import functools
import inspect
import platform
import sys
import types
from importlib import import_module
from typing import List, TypeVar
import distutils.filelist
def patch_all():
    import setuptools
    distutils.core.Command = setuptools.Command
    _patch_distribution_metadata()
    for module in (distutils.dist, distutils.core, distutils.cmd):
        module.Distribution = setuptools.dist.Distribution
    distutils.core.Extension = setuptools.extension.Extension
    distutils.extension.Extension = setuptools.extension.Extension
    if 'distutils.command.build_ext' in sys.modules:
        sys.modules['distutils.command.build_ext'].Extension = setuptools.extension.Extension
    patch_for_msvc_specialized_compiler()