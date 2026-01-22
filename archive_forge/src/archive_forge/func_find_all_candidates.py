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
@functools.lru_cache(maxsize=None)
def find_all_candidates(self, project_name: str) -> List[InstallationCandidate]:
    """Find all available InstallationCandidate for project_name

        This checks index_urls and find_links.
        All versions found are returned as an InstallationCandidate list.

        See LinkEvaluator.evaluate_link() for details on which files
        are accepted.
        """
    link_evaluator = self.make_link_evaluator(project_name)
    collected_sources = self._link_collector.collect_sources(project_name=project_name, candidates_from_page=functools.partial(self.process_project_url, link_evaluator=link_evaluator))
    page_candidates_it = itertools.chain.from_iterable((source.page_candidates() for sources in collected_sources for source in sources if source is not None))
    page_candidates = list(page_candidates_it)
    file_links_it = itertools.chain.from_iterable((source.file_links() for sources in collected_sources for source in sources if source is not None))
    file_candidates = self.evaluate_links(link_evaluator, sorted(file_links_it, reverse=True))
    if logger.isEnabledFor(logging.DEBUG) and file_candidates:
        paths = []
        for candidate in file_candidates:
            assert candidate.link.url
            try:
                paths.append(candidate.link.file_path)
            except Exception:
                paths.append(candidate.link.url)
        logger.debug('Local files found: %s', ', '.join(paths))
    return file_candidates + page_candidates