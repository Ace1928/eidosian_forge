from eidosian_core import eidosian
"""
Crawl Forge - Intelligent and ethical data harvesting.
Provides structured extraction with respect for boundaries.
"""
import requests
import time
import logging
from typing import Dict, Any, List, Optional
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse
from bs4 import BeautifulSoup

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
    def __init__(
        self,
        user_agent: str = "EidosianCrawler/0.1.0",
        *,
        enable_http_cache: bool = True,
        http_cache_ttl_seconds: float = 120.0,
        robots_cache_ttl_seconds: float = 1800.0,
    ):
        self.user_agent = user_agent
        self.rate_limit = 1.0 # Seconds between requests
        self._last_request_time = 0.0
        self._robot_parsers: Dict[str, RobotFileParser] = {}
        self._robot_parser_ts: Dict[str, float] = {}
        self.enable_http_cache = bool(enable_http_cache)
        self.http_cache_ttl_seconds = max(0.0, float(http_cache_ttl_seconds))
        self.robots_cache_ttl_seconds = max(0.0, float(robots_cache_ttl_seconds))
        self._page_cache: Dict[str, Dict[str, Any]] = {}

    def _get_robot_parser(self, url: str) -> RobotFileParser:
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        now = time.time()
        last = float(self._robot_parser_ts.get(base_url) or 0.0)
        stale = (now - last) > self.robots_cache_ttl_seconds
        if base_url not in self._robot_parsers or stale:
            rp = RobotFileParser()
            rp.set_url(f"{base_url}/robots.txt")
            try:
                rp.read()
            except Exception:
                pass
            self._robot_parsers[base_url] = rp
            self._robot_parser_ts[base_url] = now
        return self._robot_parsers[base_url]

    @eidosian()
    def can_fetch(self, url: str) -> bool:
        """Check if robots.txt allows fetching the URL."""
        rp = self._get_robot_parser(url)
        return rp.can_fetch(self.user_agent, url)

    @eidosian()
    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch a page with respect to rate limits and robots.txt."""
        if self.enable_http_cache:
            cached = self._page_cache.get(url)
            if cached:
                age = time.time() - float(cached.get("fetched_at") or 0.0)
                if age <= self.http_cache_ttl_seconds:
                    return str(cached.get("content") or "")

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
            text = response.text
            if self.enable_http_cache:
                self._page_cache[url] = {
                    "content": text,
                    "fetched_at": self._last_request_time,
                }
            return text
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            return None

    @eidosian()
    def cache_stats(self) -> Dict[str, Any]:
        """Return crawl cache status and simple counters."""
        return {
            "http_cache_enabled": self.enable_http_cache,
            "http_cache_items": len(self._page_cache),
            "http_cache_ttl_seconds": self.http_cache_ttl_seconds,
            "robots_cache_items": len(self._robot_parsers),
            "robots_cache_ttl_seconds": self.robots_cache_ttl_seconds,
        }

    @eidosian()
    def extract_structured_data(self, html: str) -> Dict[str, Any]:
        """
        Extract basic metadata from HTML using BeautifulSoup.
        """
        data = {
            "title": "",
            "meta_description": "",
            "links": []
        }

        soup = BeautifulSoup(html or "", "html.parser")

        if soup.title and soup.title.string:
            data["title"] = soup.title.string.strip()

        meta_tag = soup.find("meta", attrs={"name": lambda value: str(value).lower() == "description"})
        if meta_tag is None:
            meta_tag = soup.find("meta", attrs={"property": lambda value: str(value).lower() == "og:description"})
        if meta_tag is not None:
            data["meta_description"] = str(meta_tag.get("content") or "").strip()

        links: List[str] = []
        seen: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            href = str(anchor.get("href") or "").strip()
            if not href or href.startswith("#"):
                continue
            if not href.lower().startswith(("http://", "https://")):
                continue
            if href in seen:
                continue
            seen.add(href)
            links.append(href)
        data["links"] = links

        return data
