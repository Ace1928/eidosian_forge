import logging
import os
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Mapping, Optional, Set, Union
from ..errors import FileError, InvalidConfigError
from ..warnings import SetuptoolsWarning
from . import expand as _expand
from ._apply_pyprojecttoml import _PREVIOUSLY_DEFINED, _MissingDynamic
from ._apply_pyprojecttoml import apply as _apply
def read_configuration(filepath: _Path, expand=True, ignore_option_errors=False, dist: Optional['Distribution']=None):
    """Read given configuration file and returns options from it as a dict.

    :param str|unicode filepath: Path to configuration file in the ``pyproject.toml``
        format.

    :param bool expand: Whether to expand directives and other computed values
        (i.e. post-process the given configuration)

    :param bool ignore_option_errors: Whether to silently ignore
        options, values of which could not be resolved (e.g. due to exceptions
        in directives such as file:, attr:, etc.).
        If False exceptions are propagated as expected.

    :param Distribution|None: Distribution object to which the configuration refers.
        If not given a dummy object will be created and discarded after the
        configuration is read. This is used for auto-discovery of packages and in the
        case a dynamic configuration (e.g. ``attr`` or ``cmdclass``) is expanded.
        When ``expand=False`` this object is simply ignored.

    :rtype: dict
    """
    filepath = os.path.abspath(filepath)
    if not os.path.isfile(filepath):
        raise FileError(f'Configuration file {filepath!r} does not exist.')
    asdict = load_file(filepath) or {}
    project_table = asdict.get('project', {})
    tool_table = asdict.get('tool', {})
    setuptools_table = tool_table.get('setuptools', {})
    if not asdict or not (project_table or setuptools_table):
        return {}
    if 'distutils' in tool_table:
        _ExperimentalConfiguration.emit(subject='[tool.distutils]')
    if dist and getattr(dist, 'include_package_data', None) is not None:
        setuptools_table.setdefault('include-package-data', dist.include_package_data)
    else:
        setuptools_table.setdefault('include-package-data', True)
    asdict['tool'] = tool_table
    tool_table['setuptools'] = setuptools_table
    with _ignore_errors(ignore_option_errors):
        subset = {'project': project_table, 'tool': {'setuptools': setuptools_table}}
        validate(subset, filepath)
    if expand:
        root_dir = os.path.dirname(filepath)
        return expand_configuration(asdict, root_dir, ignore_option_errors, dist)
    return asdict