import enum
import functools
import itertools
import logging
import re
from typing import TYPE_CHECKING, FrozenSet, Iterable, List, Optional, Set, Tuple, Union
from pip._vendor.packaging import specifiers
from pip._vendor.packaging.tags import Tag
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import _BaseVersion
from pip._vendor.packaging.version import parse as parse_version
from pip._internal.exceptions import (
from pip._internal.index.collector import LinkCollector, parse_links
from pip._internal.models.candidate import InstallationCandidate
from pip._internal.models.format_control import FormatControl
from pip._internal.models.link import Link
from pip._internal.models.search_scope import SearchScope
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.models.target_python import TargetPython
from pip._internal.models.wheel import Wheel
from pip._internal.req import InstallRequirement
from pip._internal.utils._log import getLogger
from pip._internal.utils.filetypes import WHEEL_EXTENSION
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import build_netloc
from pip._internal.utils.packaging import check_requires_python
from pip._internal.utils.unpacking import SUPPORTED_EXTENSIONS
def evaluate_link(self, link: Link) -> Tuple[LinkType, str]:
    """
        Determine whether a link is a candidate for installation.

        :return: A tuple (result, detail), where *result* is an enum
            representing whether the evaluation found a candidate, or the reason
            why one is not found. If a candidate is found, *detail* will be the
            candidate's version string; if one is not found, it contains the
            reason the link fails to qualify.
        """
    version = None
    if link.is_yanked and (not self._allow_yanked):
        reason = link.yanked_reason or '<none given>'
        return (LinkType.yanked, f'yanked for reason: {reason}')
    if link.egg_fragment:
        egg_info = link.egg_fragment
        ext = link.ext
    else:
        egg_info, ext = link.splitext()
        if not ext:
            return (LinkType.format_unsupported, 'not a file')
        if ext not in SUPPORTED_EXTENSIONS:
            return (LinkType.format_unsupported, f'unsupported archive format: {ext}')
        if 'binary' not in self._formats and ext == WHEEL_EXTENSION:
            reason = f'No binaries permitted for {self.project_name}'
            return (LinkType.format_unsupported, reason)
        if 'macosx10' in link.path and ext == '.zip':
            return (LinkType.format_unsupported, 'macosx10 one')
        if ext == WHEEL_EXTENSION:
            try:
                wheel = Wheel(link.filename)
            except InvalidWheelFilename:
                return (LinkType.format_invalid, 'invalid wheel filename')
            if canonicalize_name(wheel.name) != self._canonical_name:
                reason = f'wrong project name (not {self.project_name})'
                return (LinkType.different_project, reason)
            supported_tags = self._target_python.get_unsorted_tags()
            if not wheel.supported(supported_tags):
                file_tags = ', '.join(wheel.get_formatted_file_tags())
                reason = f"none of the wheel's tags ({file_tags}) are compatible (run pip debug --verbose to show compatible tags)"
                return (LinkType.platform_mismatch, reason)
            version = wheel.version
    if 'source' not in self._formats and ext != WHEEL_EXTENSION:
        reason = f'No sources permitted for {self.project_name}'
        return (LinkType.format_unsupported, reason)
    if not version:
        version = _extract_version_from_fragment(egg_info, self._canonical_name)
    if not version:
        reason = f'Missing project version for {self.project_name}'
        return (LinkType.format_invalid, reason)
    match = self._py_version_re.search(version)
    if match:
        version = version[:match.start()]
        py_version = match.group(1)
        if py_version != self._target_python.py_version:
            return (LinkType.platform_mismatch, 'Python version is incorrect')
    supports_python = _check_link_requires_python(link, version_info=self._target_python.py_version_info, ignore_requires_python=self._ignore_requires_python)
    if not supports_python:
        reason = f'{version} Requires-Python {link.requires_python}'
        return (LinkType.requires_python_mismatch, reason)
    logger.debug('Found link %s, version: %s', link, version)
    return (LinkType.candidate, version)