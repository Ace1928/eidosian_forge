import contextlib
import functools
import os
from collections import defaultdict
from functools import partial
from functools import wraps
from typing import (
from ..errors import FileError, OptionError
from ..extern.packaging.markers import default_environment as marker_env
from ..extern.packaging.requirements import InvalidRequirement, Requirement
from ..extern.packaging.specifiers import SpecifierSet
from ..extern.packaging.version import InvalidVersion, Version
from ..warnings import SetuptoolsDeprecationWarning
from . import expand
def parse_configuration(distribution: 'Distribution', command_options: AllCommandOptions, ignore_option_errors=False) -> Tuple['ConfigMetadataHandler', 'ConfigOptionsHandler']:
    """Performs additional parsing of configuration options
    for a distribution.

    Returns a list of used option handlers.

    :param Distribution distribution:
    :param dict command_options:
    :param bool ignore_option_errors: Whether to silently ignore
        options, values of which could not be resolved (e.g. due to exceptions
        in directives such as file:, attr:, etc.).
        If False exceptions are propagated as expected.
    :rtype: list
    """
    with expand.EnsurePackagesDiscovered(distribution) as ensure_discovered:
        options = ConfigOptionsHandler(distribution, command_options, ignore_option_errors, ensure_discovered)
        options.parse()
        if not distribution.package_dir:
            distribution.package_dir = options.package_dir
        meta = ConfigMetadataHandler(distribution.metadata, command_options, ignore_option_errors, ensure_discovered, distribution.package_dir, distribution.src_root)
        meta.parse()
        distribution._referenced_files.update(options._referenced_files, meta._referenced_files)
    return (meta, options)