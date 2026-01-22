import itertools
import logging
import os
import posixpath
import urllib.parse
from typing import List
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.models.index import PyPI
from pip._internal.utils.compat import has_tls
from pip._internal.utils.misc import normalize_path, redact_auth_from_url
def get_formatted_locations(self) -> str:
    lines = []
    redacted_index_urls = []
    if self.index_urls and self.index_urls != [PyPI.simple_url]:
        for url in self.index_urls:
            redacted_index_url = redact_auth_from_url(url)
            purl = urllib.parse.urlsplit(redacted_index_url)
            if not purl.scheme and (not purl.netloc):
                logger.warning('The index url "%s" seems invalid, please provide a scheme.', redacted_index_url)
            redacted_index_urls.append(redacted_index_url)
        lines.append('Looking in indexes: {}'.format(', '.join(redacted_index_urls)))
    if self.find_links:
        lines.append('Looking in links: {}'.format(', '.join((redact_auth_from_url(url) for url in self.find_links))))
    return '\n'.join(lines)