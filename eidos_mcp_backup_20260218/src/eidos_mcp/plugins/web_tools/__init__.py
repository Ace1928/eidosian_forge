"""
🌐 Web Tools Plugin for MCP

Provides tools for web crawling, document processing, and HTTP operations.
Uses Apache Tika for document parsing.
Includes intelligent caching layer for performance.
Includes rate limiting per domain.

Created: 2026-01-23
Updated: 2026-01-23 - Added caching layer
Updated: 2026-01-23 - Added rate limiting
"""

from __future__ import annotations
from eidosian_core import eidosian

import hashlib
import json
import sqlite3
import sys
import threading
import time
import urllib.request
import urllib.error
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

PLUGIN_MANIFEST = {
    "id": "web_tools",
    "name": "Web Tools",
    "version": "1.2.0",  # Version bump for rate limiting
    "description": "Web crawling, HTTP requests, document processing with Tika (caching + rate limiting)",
    "author": "Eidos",
    "tools": [
        "web_fetch",
        "web_parse_document",
        "web_extract_links",
        "web_download",
        "web_hash_content",
        "web_cache_stats",
        "web_cache_clear",
        "web_rate_limit_status"
    ]
}


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """
    Token bucket rate limiter per domain.
    Prevents overwhelming servers with requests.
    """
    
    def __init__(
        self,
        default_rate: float = 2.0,  # requests per second
        burst_size: int = 5,        # max burst
        domain_rates: Optional[Dict[str, float]] = None
    ):
        self._lock = threading.Lock()
        self._tokens: Dict[str, float] = defaultdict(lambda: burst_size)
        self._last_update: Dict[str, float] = defaultdict(time.time)
        self.default_rate = default_rate
        self.burst_size = burst_size
        self.domain_rates = domain_rates or {}
        self._request_counts: Dict[str, int] = defaultdict(int)
    
    def _get_rate(self, domain: str) -> float:
        """Get rate limit for domain."""
        return self.domain_rates.get(domain, self.default_rate)
    
    @eidosian()
    def acquire(self, url: str, timeout: float = 30.0) -> bool:
        """
        Acquire permission to make a request.
        Blocks until allowed or timeout.
        
        Returns:
            True if acquired, False if timeout
        """
        domain = urlparse(url).netloc
        rate = self._get_rate(domain)
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            with self._lock:
                now = time.time()
                elapsed = now - self._last_update[domain]
                self._last_update[domain] = now
                
                # Add tokens based on time elapsed
                self._tokens[domain] = min(
                    self.burst_size,
                    self._tokens[domain] + elapsed * rate
                )
                
                # Try to consume a token
                if self._tokens[domain] >= 1.0:
                    self._tokens[domain] -= 1.0
                    self._request_counts[domain] += 1
                    return True
            
            # Wait a bit before retry
            time.sleep(0.1)
        
        return False
    
    @eidosian()
    def status(self) -> Dict[str, Any]:
        """Get rate limiter status."""
        with self._lock:
            return {
                "default_rate": self.default_rate,
                "burst_size": self.burst_size,
                "domains": {
                    domain: {
                        "tokens": round(self._tokens[domain], 2),
                        "rate": self._get_rate(domain),
                        "requests_made": self._request_counts[domain]
                    }
                    for domain in self._tokens.keys()
                },
                "custom_rates": self.domain_rates
            }


# Global rate limiter
_rate_limiter = RateLimiter(
    default_rate=2.0,  # 2 requests/second default
    burst_size=5,      # Allow bursts of 5
    domain_rates={
        "api.github.com": 1.0,      # GitHub rate limit
        "httpbin.org": 5.0,         # Permissive test server
    }
)


# ============================================================================
# CACHING LAYER
# ============================================================================

class WebCache:
    """
    Thread-safe URL content cache with TTL support.
    Uses SQLite for persistence across sessions.
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        default_ttl: int = 3600,  # 1 hour
        max_size_mb: int = 100
    ):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "eidosian_web"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "cache.db"
        self.default_ttl = default_ttl
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = threading.Lock()
        self._init_db()
        
        # Stats
        self.hits = 0
        self.misses = 0
    
    def _init_db(self) -> None:
        """Initialize cache database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    url TEXT PRIMARY KEY,
                    content_hash TEXT,
                    content BLOB,
                    content_type TEXT,
                    headers TEXT,
                    status_code INTEGER,
                    cached_at REAL,
                    expires_at REAL,
                    hit_count INTEGER DEFAULT 0,
                    size_bytes INTEGER
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires 
                ON cache(expires_at)
            """)
            conn.commit()
    
    @eidosian()
    def get(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached content if valid."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cur = conn.execute(
                        "SELECT * FROM cache WHERE url = ? AND expires_at > ?",
                        (url, time.time())
                    )
                    row = cur.fetchone()
                    
                    if row:
                        # Update hit count
                        conn.execute(
                            "UPDATE cache SET hit_count = hit_count + 1 WHERE url = ?",
                            (url,)
                        )
                        conn.commit()
                        self.hits += 1
                        
                        return {
                            "url": row["url"],
                            "content": row["content"],
                            "content_type": row["content_type"],
                            "content_hash": row["content_hash"],
                            "headers": json.loads(row["headers"]) if row["headers"] else {},
                            "status_code": row["status_code"],
                            "cached_at": row["cached_at"],
                            "from_cache": True
                        }
                    
                    self.misses += 1
                    return None
                    
            except Exception:
                self.misses += 1
                return None
    
    @eidosian()
    def put(
        self,
        url: str,
        content: bytes,
        content_type: str,
        headers: Dict[str, str],
        status_code: int,
        ttl: Optional[int] = None
    ) -> None:
        """Store content in cache."""
        ttl = ttl or self.default_ttl
        now = time.time()
        content_hash = hashlib.sha256(content).hexdigest()[:16]
        
        with self._lock:
            try:
                # Check size limit and evict if needed
                self._evict_if_needed(len(content))
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache 
                        (url, content_hash, content, content_type, headers, 
                         status_code, cached_at, expires_at, size_bytes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        url, content_hash, content, content_type,
                        json.dumps(headers), status_code,
                        now, now + ttl, len(content)
                    ))
                    conn.commit()
            except Exception:
                pass  # Cache failures shouldn't break functionality
    
    def _evict_if_needed(self, new_size: int) -> None:
        """Evict old entries if cache is too large."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get current size
                cur = conn.execute("SELECT SUM(size_bytes) FROM cache")
                current_size = cur.fetchone()[0] or 0
                
                # Evict expired first
                conn.execute("DELETE FROM cache WHERE expires_at < ?", (time.time(),))
                
                # If still too large, evict LRU
                if current_size + new_size > self.max_size_bytes:
                    # Delete oldest 20%
                    conn.execute("""
                        DELETE FROM cache WHERE url IN (
                            SELECT url FROM cache 
                            ORDER BY hit_count ASC, cached_at ASC 
                            LIMIT (SELECT COUNT(*) / 5 FROM cache)
                        )
                    """)
                
                conn.commit()
        except Exception:
            pass
    
    @eidosian()
    def clear(self, older_than: Optional[int] = None) -> int:
        """Clear cache entries."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    if older_than:
                        threshold = time.time() - older_than
                        cur = conn.execute(
                            "DELETE FROM cache WHERE cached_at < ?", (threshold,)
                        )
                    else:
                        cur = conn.execute("DELETE FROM cache")
                    conn.commit()
                    return cur.rowcount
            except Exception:
                return 0
    
    @eidosian()
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    
                    # Total entries
                    cur = conn.execute("SELECT COUNT(*) as count FROM cache")
                    total = cur.fetchone()["count"]
                    
                    # Total size
                    cur = conn.execute("SELECT SUM(size_bytes) as size FROM cache")
                    size = cur.fetchone()["size"] or 0
                    
                    # Valid entries
                    cur = conn.execute(
                        "SELECT COUNT(*) as count FROM cache WHERE expires_at > ?",
                        (time.time(),)
                    )
                    valid = cur.fetchone()["count"]
                    
                    # Top domains
                    cur = conn.execute("""
                        SELECT url, hit_count FROM cache 
                        ORDER BY hit_count DESC LIMIT 5
                    """)
                    top_urls = [(row["url"][:50], row["hit_count"]) for row in cur]
                    
                    return {
                        "total_entries": total,
                        "valid_entries": valid,
                        "expired_entries": total - valid,
                        "total_size_bytes": size,
                        "total_size_mb": round(size / (1024 * 1024), 2),
                        "cache_hits": self.hits,
                        "cache_misses": self.misses,
                        "hit_rate": round(self.hits / max(1, self.hits + self.misses), 3),
                        "top_cached": top_urls,
                        "cache_dir": str(self.cache_dir)
                    }
            except Exception as e:
                return {"error": str(e)}


# Global cache instance
_cache = WebCache(
    cache_dir=Path.home() / ".cache" / "eidosian_web",
    default_ttl=3600,  # 1 hour default
    max_size_mb=100
)

_TIKA_AVAILABLE: bool | None = None
_TIKA_PARSER = None


def _get_tika_parser():
    global _TIKA_AVAILABLE, _TIKA_PARSER
    if _TIKA_AVAILABLE is False:
        return None
    if _TIKA_PARSER is None:
        try:
            from tika import parser as tika_parser
            import os
            os.environ.setdefault("TIKA_LOG_PATH", "/tmp")
            _TIKA_PARSER = tika_parser
            _TIKA_AVAILABLE = True
        except ImportError:
            _TIKA_AVAILABLE = False
            return None
    return _TIKA_PARSER


@eidosian()
def web_fetch(
    url: str,
    timeout: int = 30,
    headers: Optional[Dict[str, str]] = None,
    use_cache: bool = True,
    cache_ttl: Optional[int] = None,
    respect_rate_limit: bool = True
) -> str:
    """
    Fetch content from a URL.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        headers: Optional HTTP headers
        use_cache: Whether to use cached content (default: True)
        cache_ttl: Cache TTL in seconds (default: 3600)
        respect_rate_limit: Whether to apply rate limiting (default: True)
    
    Returns:
        JSON string with content and metadata
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return json.dumps({"status": "error", "error": "Only HTTP/HTTPS URLs supported"})
        
        # Check cache first (before rate limiting)
        if use_cache:
            cached = _cache.get(url)
            if cached:
                try:
                    text_content = cached["content"].decode('utf-8')
                except (UnicodeDecodeError, AttributeError):
                    text_content = None
                
                return json.dumps({
                    "status": "success",
                    "url": url,
                    "status_code": cached["status_code"],
                    "content_type": cached["content_type"],
                    "content_length": len(cached["content"]),
                    "content_hash": cached["content_hash"],
                    "content": text_content[:10000] if text_content else f"[Binary content: {len(cached['content'])} bytes]",
                    "headers": cached["headers"],
                    "from_cache": True,
                    "rate_limited": False,
                    "cached_at": datetime.fromtimestamp(cached["cached_at"], tz=timezone.utc).isoformat(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }, indent=2)
        
        # Apply rate limiting for non-cached requests
        if respect_rate_limit:
            if not _rate_limiter.acquire(url, timeout=min(timeout, 10)):
                return json.dumps({
                    "status": "error",
                    "url": url,
                    "error": "Rate limit exceeded - try again later",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Eidosian-MCP/1.2')
        
        if headers:
            for key, value in headers.items():
                req.add_header(key, value)
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read()
            content_type = response.headers.get('Content-Type', 'unknown')
            response_headers = dict(response.headers)
            status_code = response.status
            
            # Store in cache
            if use_cache:
                _cache.put(
                    url, content, content_type, 
                    response_headers, status_code, 
                    ttl=cache_ttl
                )
            
            # Try to decode as text
            try:
                text_content = content.decode('utf-8')
            except UnicodeDecodeError:
                text_content = None
            
            return json.dumps({
                "status": "success",
                "url": url,
                "status_code": status_code,
                "content_type": content_type,
                "content_length": len(content),
                "content_hash": hashlib.sha256(content).hexdigest()[:16],
                "content": text_content[:10000] if text_content else f"[Binary content: {len(content)} bytes]",
                "headers": response_headers,
                "from_cache": False,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, indent=2)
            
    except urllib.error.HTTPError as e:
        return json.dumps({
            "status": "error",
            "url": url,
            "error": f"HTTP {e.code}: {e.reason}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "url": url,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


@eidosian()
def web_parse_document(
    source: str,
    source_type: str = "auto"
) -> str:
    """
    Parse a document using Apache Tika.
    
    Args:
        source: URL or file path to parse
        source_type: 'url', 'file', or 'auto' (detect automatically)
    
    Returns:
        JSON string with parsed content and metadata
    """
    tika_parser = _get_tika_parser()
    if not tika_parser:
        return json.dumps({
            "status": "error",
            "error": "Tika not available. Install with: pip install tika"
        })
    
    try:
        # Determine source type
        if source_type == "auto":
            if source.startswith(('http://', 'https://')):
                source_type = "url"
            else:
                source_type = "file"
        
        # Parse based on type
        if source_type == "url":
            parsed = tika_parser.from_url(source)
        else:
            if not Path(source).exists():
                return json.dumps({"status": "error", "error": f"File not found: {source}"})
            parsed = tika_parser.from_file(source)
        
        content = parsed.get("content", "")
        metadata = parsed.get("metadata", {})
        
        return json.dumps({
            "status": "success",
            "source": source,
            "source_type": source_type,
            "content_type": metadata.get("Content-Type", "unknown"),
            "content_length": len(content) if content else 0,
            "content": content[:15000] if content else "",
            "metadata": {k: v for k, v in metadata.items() if not k.startswith("X-")},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "source": source,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


@eidosian()
def web_extract_links(url: str, timeout: int = 30) -> str:
    """
    Extract all links from a webpage.
    
    Args:
        url: URL to extract links from
        timeout: Request timeout
    
    Returns:
        JSON string with extracted links
    """
    import re
    
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Eidosian-MCP/1.0')
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read().decode('utf-8', errors='ignore')
        
        # Extract href links
        href_pattern = r'href=["\']([^"\']+)["\']'
        links = re.findall(href_pattern, content, re.IGNORECASE)
        
        # Resolve relative URLs
        from urllib.parse import urljoin
        resolved_links = []
        for link in links:
            if link.startswith(('http://', 'https://')):
                resolved_links.append(link)
            elif link.startswith('/'):
                resolved_links.append(urljoin(url, link))
            elif not link.startswith(('#', 'javascript:', 'mailto:')):
                resolved_links.append(urljoin(url, link))
        
        # Deduplicate while preserving order
        seen = set()
        unique_links = []
        for link in resolved_links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
        
        return json.dumps({
            "status": "success",
            "source_url": url,
            "total_links": len(unique_links),
            "links": unique_links[:100],  # Limit to first 100
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "url": url,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


@eidosian()
def web_download(
    url: str,
    save_path: str,
    timeout: int = 60
) -> str:
    """
    Download a file from URL.
    
    Args:
        url: URL to download
        save_path: Path to save the file
        timeout: Request timeout
    
    Returns:
        JSON string with download result
    """
    try:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Eidosian-MCP/1.0')
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read()
            content_type = response.headers.get('Content-Type', 'unknown')
            
            with open(save_path, 'wb') as f:
                f.write(content)
        
        return json.dumps({
            "status": "success",
            "url": url,
            "saved_to": str(save_path),
            "size_bytes": len(content),
            "content_type": content_type,
            "content_hash": hashlib.sha256(content).hexdigest(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "url": url,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


@eidosian()
def web_hash_content(url: str, algorithm: str = "sha256") -> str:
    """
    Compute hash of content at URL without storing it.
    
    Args:
        url: URL to hash
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)
    
    Returns:
        JSON string with hash
    """
    algorithms = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512
    }
    
    if algorithm not in algorithms:
        return json.dumps({
            "status": "error",
            "error": f"Unsupported algorithm. Choose from: {list(algorithms.keys())}"
        })
    
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Eidosian-MCP/1.1')
        
        hasher = algorithms[algorithm]()
        
        with urllib.request.urlopen(req, timeout=60) as response:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)
        
        return json.dumps({
            "status": "success",
            "url": url,
            "algorithm": algorithm,
            "hash": hasher.hexdigest(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "url": url,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


@eidosian()
def web_cache_stats() -> str:
    """
    Get cache statistics.
    
    Returns:
        JSON string with cache stats including hit rate, size, entries
    """
    stats = _cache.stats()
    return json.dumps({
        "status": "success",
        **stats,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }, indent=2)


@eidosian()
def web_cache_clear(older_than_hours: Optional[int] = None) -> str:
    """
    Clear cache entries.
    
    Args:
        older_than_hours: If specified, only clear entries older than this many hours
    
    Returns:
        JSON string with number of entries cleared
    """
    older_than = older_than_hours * 3600 if older_than_hours else None
    cleared = _cache.clear(older_than)
    
    return json.dumps({
        "status": "success",
        "entries_cleared": cleared,
        "filter": f"older than {older_than_hours} hours" if older_than_hours else "all",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


@eidosian()
def web_rate_limit_status() -> str:
    """
    Get rate limiter status per domain.
    
    Returns:
        JSON string with rate limiter status including:
        - Default rate and burst size
        - Per-domain token counts and request counts
        - Custom rate limits
    """
    status = _rate_limiter.status()
    return json.dumps({
        "status": "success",
        **status,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }, indent=2)
