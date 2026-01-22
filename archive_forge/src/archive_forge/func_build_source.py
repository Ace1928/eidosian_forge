import logging
import mimetypes
import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from pip._vendor.packaging.utils import (
from pip._internal.models.candidate import InstallationCandidate
from pip._internal.models.link import Link
from pip._internal.utils.urls import path_to_url, url_to_path
from pip._internal.vcs import is_url
def build_source(location: str, *, candidates_from_page: CandidatesFromPage, page_validator: PageValidator, expand_dir: bool, cache_link_parsing: bool, project_name: str) -> Tuple[Optional[str], Optional[LinkSource]]:
    path: Optional[str] = None
    url: Optional[str] = None
    if os.path.exists(location):
        url = path_to_url(location)
        path = location
    elif location.startswith('file:'):
        url = location
        path = url_to_path(location)
    elif is_url(location):
        url = location
    if url is None:
        msg = "Location '%s' is ignored: it is either a non-existing path or lacks a specific scheme."
        logger.warning(msg, location)
        return (None, None)
    if path is None:
        source: LinkSource = _RemoteFileSource(candidates_from_page=candidates_from_page, page_validator=page_validator, link=Link(url, cache_link_parsing=cache_link_parsing))
        return (url, source)
    if os.path.isdir(path):
        if expand_dir:
            source = _FlatDirectorySource(candidates_from_page=candidates_from_page, path=path, project_name=project_name)
        else:
            source = _IndexDirectorySource(candidates_from_page=candidates_from_page, link=Link(url, cache_link_parsing=cache_link_parsing))
        return (url, source)
    elif os.path.isfile(path):
        source = _LocalFileSource(candidates_from_page=candidates_from_page, link=Link(url, cache_link_parsing=cache_link_parsing))
        return (url, source)
    logger.warning("Location '%s' is ignored: it is neither a file nor a directory.", location)
    return (url, None)