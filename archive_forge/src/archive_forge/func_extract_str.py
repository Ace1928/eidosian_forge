from __future__ import annotations
import logging
import os
import urllib.parse
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import wraps
import idna
import requests
from .cache import DiskCache, get_cache_dir
from .remote import lenient_netloc, looks_like_ip, looks_like_ipv6
from .suffix_list import get_suffix_lists
def extract_str(self, url: str, include_psl_private_domains: bool | None=None, session: requests.Session | None=None) -> ExtractResult:
    """Take a string URL and splits it into its subdomain, domain, and suffix components.

        I.e. its effective TLD, gTLD, ccTLD, etc. components.

        >>> extractor = TLDExtract()
        >>> extractor.extract_str('http://forums.news.cnn.com/')
        ExtractResult(subdomain='forums.news', domain='cnn', suffix='com', is_private=False)
        >>> extractor.extract_str('http://forums.bbc.co.uk/')
        ExtractResult(subdomain='forums', domain='bbc', suffix='co.uk', is_private=False)

        Allows configuring the HTTP request via the optional `session`
        parameter. For example, if you need to use a HTTP proxy. See also
        `requests.Session`.

        >>> import requests
        >>> session = requests.Session()
        >>> # customize your session here
        >>> with session:
        ...     extractor.extract_str("http://forums.news.cnn.com/", session=session)
        ExtractResult(subdomain='forums.news', domain='cnn', suffix='com', is_private=False)
        """
    return self._extract_netloc(lenient_netloc(url), include_psl_private_domains, session=session)