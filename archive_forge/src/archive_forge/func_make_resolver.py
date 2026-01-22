import logging
import os
import sys
from functools import partial
from optparse import Values
from typing import TYPE_CHECKING, Any, List, Optional, Tuple
from pip._internal.cache import WheelCache
from pip._internal.cli import cmdoptions
from pip._internal.cli.base_command import Command
from pip._internal.cli.command_context import CommandContextMixIn
from pip._internal.exceptions import CommandError, PreviousBuildDirError
from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.models.target_python import TargetPython
from pip._internal.network.session import PipSession
from pip._internal.operations.build.build_tracker import BuildTracker
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req.constructors import (
from pip._internal.req.req_file import parse_requirements
from pip._internal.req.req_install import InstallRequirement
from pip._internal.resolution.base import BaseResolver
from pip._internal.self_outdated_check import pip_self_version_check
from pip._internal.utils.temp_dir import (
from pip._internal.utils.virtualenv import running_under_virtualenv
@classmethod
def make_resolver(cls, preparer: RequirementPreparer, finder: PackageFinder, options: Values, wheel_cache: Optional[WheelCache]=None, use_user_site: bool=False, ignore_installed: bool=True, ignore_requires_python: bool=False, force_reinstall: bool=False, upgrade_strategy: str='to-satisfy-only', use_pep517: Optional[bool]=None, py_version_info: Optional[Tuple[int, ...]]=None) -> BaseResolver:
    """
        Create a Resolver instance for the given parameters.
        """
    make_install_req = partial(install_req_from_req_string, isolated=options.isolated_mode, use_pep517=use_pep517)
    resolver_variant = cls.determine_resolver_variant(options)
    if resolver_variant == 'resolvelib':
        import pip._internal.resolution.resolvelib.resolver
        return pip._internal.resolution.resolvelib.resolver.Resolver(preparer=preparer, finder=finder, wheel_cache=wheel_cache, make_install_req=make_install_req, use_user_site=use_user_site, ignore_dependencies=options.ignore_dependencies, ignore_installed=ignore_installed, ignore_requires_python=ignore_requires_python, force_reinstall=force_reinstall, upgrade_strategy=upgrade_strategy, py_version_info=py_version_info)
    import pip._internal.resolution.legacy.resolver
    return pip._internal.resolution.legacy.resolver.Resolver(preparer=preparer, finder=finder, wheel_cache=wheel_cache, make_install_req=make_install_req, use_user_site=use_user_site, ignore_dependencies=options.ignore_dependencies, ignore_installed=ignore_installed, ignore_requires_python=ignore_requires_python, force_reinstall=force_reinstall, upgrade_strategy=upgrade_strategy, py_version_info=py_version_info)