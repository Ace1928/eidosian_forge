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
def get_applicable_candidates(self, candidates: List[InstallationCandidate]) -> List[InstallationCandidate]:
    """
        Return the applicable candidates from a list of candidates.
        """
    allow_prereleases = self._allow_all_prereleases or None
    specifier = self._specifier
    versions = {str(v) for v in specifier.filter((str(c.version) for c in candidates), prereleases=allow_prereleases)}
    applicable_candidates = [c for c in candidates if str(c.version) in versions]
    filtered_applicable_candidates = filter_unallowed_hashes(candidates=applicable_candidates, hashes=self._hashes, project_name=self._project_name)
    return sorted(filtered_applicable_candidates, key=self._sort_key)