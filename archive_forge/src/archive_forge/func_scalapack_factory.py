from __future__ import annotations
from pathlib import Path
import functools
import os
import typing as T
from ..mesonlib import OptionKey
from .base import DependencyMethods
from .cmake import CMakeDependency
from .detect import packages
from .pkgconfig import PkgConfigDependency
from .factory import factory_methods
@factory_methods({DependencyMethods.PKGCONFIG, DependencyMethods.CMAKE})
def scalapack_factory(env: 'Environment', for_machine: 'MachineChoice', kwargs: T.Dict[str, T.Any], methods: T.List[DependencyMethods]) -> T.List['DependencyGenerator']:
    candidates: T.List['DependencyGenerator'] = []
    if DependencyMethods.PKGCONFIG in methods:
        static_opt = kwargs.get('static', env.coredata.get_option(OptionKey('prefer_static')))
        mkl = 'mkl-static-lp64-iomp' if static_opt else 'mkl-dynamic-lp64-iomp'
        candidates.append(functools.partial(MKLPkgConfigDependency, mkl, env, kwargs))
        for pkg in ['scalapack-openmpi', 'scalapack']:
            candidates.append(functools.partial(PkgConfigDependency, pkg, env, kwargs))
    if DependencyMethods.CMAKE in methods:
        candidates.append(functools.partial(CMakeDependency, 'Scalapack', env, kwargs))
    return candidates