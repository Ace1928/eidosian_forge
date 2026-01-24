from eidosian_core import eidosian
"""
Crawl Forge - Intelligent and ethical data harvesting.
Provides structured extraction with respect for boundaries.
"""
import requests
import time
import logging
import re
from typing import Dict, Any, List, Optional
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse

# Export Tika integration
try:
    from .tika_extractor import TikaExtractor, TikaKnowledgeIngester
except ImportError:
    TikaExtractor = None
    TikaKnowledgeIngester = None

__all__ = ["CrawlForge", "TikaExtractor", "TikaKnowledgeIngester"]


class CrawlForge:
    """
    Manages ethical web crawling and data extraction.
    """
    def __init__(self, user_agent: str = "EidosianCrawler/0.1.0"):
        self.user_agent = user_agent
        self.rate_limit = 1.0 # Seconds between requests
        self._last_request_time = 0.0
        self._robot_parsers: Dict[str, RobotFileParser] = {}

    def _get_robot_parser(self, url: str) -> RobotFileParser:
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        if base_url not in self._robot_parsers:
            rp = RobotFileParser()
            rp.set_url(f"{base_url}/robots.txt")
            try:
                rp.read()
            except Exception:
                pass
            self._robot_parsers[base_url] = rp
        return self._robot_parsers[base_url]

    @eidosian()
    def can_fetch(self, url: str) -> bool:
        """Check if robots.txt allows fetching the URL."""
        rp = self._get_robot_parser(url)
        return rp.can_fetch(self.user_agent, url)

    @eidosian()
    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch a page with respect to rate limits and robots.txt."""
        if not self.can_fetch(url):
            logging.warning(f"Robots.txt forbids fetching: {url}")
            return None

        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

        try:
            response = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=10)
            self._last_request_time = time.time()
            response.raise_for_status()
            return response.text
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            return None

    @eidosian()
    def extract_structured_data(self, html: str) -> Dict[str, Any]:
        """
        Extract basic metadata from HTML using regex.
        """
        data = {
            "title": "",
            "meta_description": "",
            "links": []
        }
        
        # Extract title
        title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if title_match:
            data["title"] = title_match.group(1).strip()
            
        # Extract meta description
        meta_desc_match = re.search(r'<meta name="description" content="(.*?)"', html, re.IGNORECASE)
        if meta_desc_match:
            data["meta_description"] = meta_desc_match.group(1).strip()
            
        # Extract all links
        links = re.findall(r'href=["\'](http[s]?://.*?)["\']', html, re.IGNORECASE)
        data["links"] = list(set(links)) # Deduplicate
        
        return data
