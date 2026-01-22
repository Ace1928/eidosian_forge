import importlib.resources
import locale
import logging
import os
import sys
from optparse import Values
from types import ModuleType
from typing import Any, Dict, List, Optional
import pip._vendor
from pip._vendor.certifi import where
from pip._vendor.packaging.version import parse as parse_version
from pip._internal.cli import cmdoptions
from pip._internal.cli.base_command import Command
from pip._internal.cli.cmdoptions import make_target_python
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.configuration import Configuration
from pip._internal.metadata import get_environment
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import get_pip_version
def ca_bundle_info(config: Configuration) -> str:
    levels = {key.split('.', 1)[0] for key, _ in config.items()}
    if not levels:
        return 'Not specified'
    levels_that_override_global = ['install', 'wheel', 'download']
    global_overriding_level = [level for level in levels if level in levels_that_override_global]
    if not global_overriding_level:
        return 'global'
    if 'global' in levels:
        levels.remove('global')
    return ', '.join(levels)