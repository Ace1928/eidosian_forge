import logging
import optparse
import os
import re
import shlex
import urllib.parse
from optparse import Values
from typing import (
from pip._internal.cli import cmdoptions
from pip._internal.exceptions import InstallationError, RequirementsFileParseError
from pip._internal.models.search_scope import SearchScope
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status
from pip._internal.utils.encoding import auto_decode
from pip._internal.utils.urls import get_url_scheme
def handle_option_line(opts: Values, filename: str, lineno: int, finder: Optional['PackageFinder']=None, options: Optional[optparse.Values]=None, session: Optional[PipSession]=None) -> None:
    if opts.hashes:
        logger.warning('%s line %s has --hash but no requirement, and will be ignored.', filename, lineno)
    if options:
        if opts.require_hashes:
            options.require_hashes = opts.require_hashes
        if opts.features_enabled:
            options.features_enabled.extend((f for f in opts.features_enabled if f not in options.features_enabled))
    if finder:
        find_links = finder.find_links
        index_urls = finder.index_urls
        no_index = finder.search_scope.no_index
        if opts.no_index is True:
            no_index = True
            index_urls = []
        if opts.index_url and (not no_index):
            index_urls = [opts.index_url]
        if opts.extra_index_urls and (not no_index):
            index_urls.extend(opts.extra_index_urls)
        if opts.find_links:
            value = opts.find_links[0]
            req_dir = os.path.dirname(os.path.abspath(filename))
            relative_to_reqs_file = os.path.join(req_dir, value)
            if os.path.exists(relative_to_reqs_file):
                value = relative_to_reqs_file
            find_links.append(value)
        if session:
            session.update_index_urls(index_urls)
        search_scope = SearchScope(find_links=find_links, index_urls=index_urls, no_index=no_index)
        finder.search_scope = search_scope
        if opts.pre:
            finder.set_allow_all_prereleases()
        if opts.prefer_binary:
            finder.set_prefer_binary()
        if session:
            for host in opts.trusted_hosts or []:
                source = f'line {lineno} of {filename}'
                session.add_trusted_host(host, source=source)