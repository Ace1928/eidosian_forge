import itertools
import re
from typing import Any, Callable, Generator, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import urlparse
from langchain_core.documents import Document
from langchain_community.document_loaders.web_base import WebBaseLoader
def parse_sitemap(self, soup: Any) -> List[dict]:
    """Parse sitemap xml and load into a list of dicts.

        Args:
            soup: BeautifulSoup object.

        Returns:
            List of dicts.
        """
    els = []
    for url in soup.find_all('url'):
        loc = url.find('loc')
        if not loc:
            continue
        loc_text = loc.text.strip()
        if self.restrict_to_same_domain and (not self.is_local):
            if _extract_scheme_and_domain(loc_text) != _extract_scheme_and_domain(self.web_path):
                continue
        if self.allow_url_patterns and (not any((re.match(regexp_pattern, loc_text) for regexp_pattern in self.allow_url_patterns))):
            continue
        els.append({tag: prop.text for tag in ['loc', 'lastmod', 'changefreq', 'priority'] if (prop := url.find(tag))})
    for sitemap in soup.find_all('sitemap'):
        loc = sitemap.find('loc')
        if not loc:
            continue
        soup_child = self.scrape_all([loc.text], 'xml')[0]
        els.extend(self.parse_sitemap(soup_child))
    return els