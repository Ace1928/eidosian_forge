import functools
import os
import sys
import sysconfig
from importlib.util import cache_from_source
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple
from pip._internal.exceptions import UninstallationError
from pip._internal.locations import get_bin_prefix, get_bin_user
from pip._internal.metadata import BaseDistribution
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.egg_link import egg_link_path_from_location
from pip._internal.utils.logging import getLogger, indent_log
from pip._internal.utils.misc import ask, normalize_path, renames, rmtree
from pip._internal.utils.temp_dir import AdjacentTempDirectory, TempDirectory
from pip._internal.utils.virtualenv import running_under_virtualenv
@classmethod
def from_dist(cls, dist: BaseDistribution) -> 'UninstallPathSet':
    dist_location = dist.location
    info_location = dist.info_location
    if dist_location is None:
        logger.info('Not uninstalling %s since it is not installed', dist.canonical_name)
        return cls(dist)
    normalized_dist_location = normalize_path(dist_location)
    if not dist.local:
        logger.info('Not uninstalling %s at %s, outside environment %s', dist.canonical_name, normalized_dist_location, sys.prefix)
        return cls(dist)
    if normalized_dist_location in {p for p in {sysconfig.get_path('stdlib'), sysconfig.get_path('platstdlib')} if p}:
        logger.info('Not uninstalling %s at %s, as it is in the standard library.', dist.canonical_name, normalized_dist_location)
        return cls(dist)
    paths_to_remove = cls(dist)
    develop_egg_link = egg_link_path_from_location(dist.raw_name)
    setuptools_flat_installation = dist.installed_with_setuptools_egg_info and info_location is not None and os.path.exists(info_location) and (not info_location.endswith(f'{dist.setuptools_filename}.egg-info'))
    if setuptools_flat_installation:
        if info_location is not None:
            paths_to_remove.add(info_location)
        installed_files = dist.iter_declared_entries()
        if installed_files is not None:
            for installed_file in installed_files:
                paths_to_remove.add(os.path.join(dist_location, installed_file))
        elif dist.is_file('top_level.txt'):
            try:
                namespace_packages = dist.read_text('namespace_packages.txt')
            except FileNotFoundError:
                namespaces = []
            else:
                namespaces = namespace_packages.splitlines(keepends=False)
            for top_level_pkg in [p for p in dist.read_text('top_level.txt').splitlines() if p and p not in namespaces]:
                path = os.path.join(dist_location, top_level_pkg)
                paths_to_remove.add(path)
                paths_to_remove.add(f'{path}.py')
                paths_to_remove.add(f'{path}.pyc')
                paths_to_remove.add(f'{path}.pyo')
    elif dist.installed_by_distutils:
        raise UninstallationError('Cannot uninstall {!r}. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.'.format(dist.raw_name))
    elif dist.installed_as_egg:
        paths_to_remove.add(dist_location)
        easy_install_egg = os.path.split(dist_location)[1]
        easy_install_pth = os.path.join(os.path.dirname(dist_location), 'easy-install.pth')
        paths_to_remove.add_pth(easy_install_pth, './' + easy_install_egg)
    elif dist.installed_with_dist_info:
        for path in uninstallation_paths(dist):
            paths_to_remove.add(path)
    elif develop_egg_link:
        with open(develop_egg_link) as fh:
            link_pointer = os.path.normcase(fh.readline().strip())
            normalized_link_pointer = paths_to_remove._normalize_path_cached(link_pointer)
        assert os.path.samefile(normalized_link_pointer, normalized_dist_location), f'Egg-link {develop_egg_link} (to {link_pointer}) does not match installed location of {dist.raw_name} (at {dist_location})'
        paths_to_remove.add(develop_egg_link)
        easy_install_pth = os.path.join(os.path.dirname(develop_egg_link), 'easy-install.pth')
        paths_to_remove.add_pth(easy_install_pth, dist_location)
    else:
        logger.debug('Not sure how to uninstall: %s - Check: %s', dist, dist_location)
    if dist.in_usersite:
        bin_dir = get_bin_user()
    else:
        bin_dir = get_bin_prefix()
    try:
        for script in dist.iter_distutils_script_names():
            paths_to_remove.add(os.path.join(bin_dir, script))
            if WINDOWS:
                paths_to_remove.add(os.path.join(bin_dir, f'{script}.bat'))
    except (FileNotFoundError, NotADirectoryError):
        pass

    def iter_scripts_to_remove(dist: BaseDistribution, bin_dir: str) -> Generator[str, None, None]:
        for entry_point in dist.iter_entry_points():
            if entry_point.group == 'console_scripts':
                yield from _script_names(bin_dir, entry_point.name, False)
            elif entry_point.group == 'gui_scripts':
                yield from _script_names(bin_dir, entry_point.name, True)
    for s in iter_scripts_to_remove(dist, bin_dir):
        paths_to_remove.add(s)
    return paths_to_remove