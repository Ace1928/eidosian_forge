from __future__ import annotations
import base64
import dataclasses
import json
import os
import re
import typing as t
from .encoding import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .host_configs import (
from .connections import (
from .coverage_util import (
def collect_requirements(python: PythonConfig, controller: bool, ansible: bool, cryptography: bool, coverage: bool, virtualenv: bool, minimize: bool, command: t.Optional[str], sanity: t.Optional[str]) -> list[PipCommand]:
    """Collect requirements for the given Python using the specified arguments."""
    commands: list[PipCommand] = []
    if virtualenv:
        commands.extend(collect_package_install(packages=[f'virtualenv=={VIRTUALENV_VERSION}'], constraints=False))
    if coverage:
        commands.extend(collect_package_install(packages=[f'coverage=={get_coverage_version(python.version).coverage_version}'], constraints=False))
    if cryptography:
        commands.extend(collect_package_install(packages=get_cryptography_requirements(python)))
    if ansible or command:
        commands.extend(collect_general_install(command, ansible))
    if sanity:
        commands.extend(collect_sanity_install(sanity))
    if command == 'units':
        commands.extend(collect_units_install())
    if command in ('integration', 'windows-integration', 'network-integration'):
        commands.extend(collect_integration_install(command, controller))
    if (sanity or minimize) and any((isinstance(command, PipInstall) for command in commands)):
        commands = collect_bootstrap(python) + commands
        uninstall_packages = list(get_venv_packages(python))
        if not minimize:
            uninstall_packages.remove('setuptools')
        install_commands = [command for command in commands if isinstance(command, PipInstall)]
        install_wheel = any((install.has_package('wheel') for install in install_commands))
        if install_wheel:
            uninstall_packages.remove('wheel')
        commands.extend(collect_uninstall(packages=uninstall_packages))
    return commands