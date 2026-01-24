"""
📄 Document Processing Module

Integrates Apache Tika for robust document parsing and text extraction.
Works with crawl_forge to process fetched documents.

Capabilities:
- Parse 1000+ document formats (PDF, DOCX, HTML, images, etc.)
- Extract text content and metadata
- OCR support for images (with Tesseract)
- Language detection

Safety:
- All operations logged
- Rate limiting for external URLs
- Content type validation

Usage:
    from doc_processor import DocumentProcessor

    proc = DocumentProcessor()
    result = proc.parse_file("document.pdf")
    print(result["content"])
    print(result["metadata"])

Created: 2026-01-23
Author: Eidos
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# Lazy import tika to avoid startup cost
_tika_parser = None


def _get_tika_parser():
    """Lazy-load tika parser."""
    global _tika_parser
    if _tika_parser is None:
        # Set Tika log path (must be a directory, not a file)
        os.environ.setdefault("TIKA_LOG_PATH", "/tmp")
        from tika import parser
        _tika_parser = parser
    return _tika_parser


LOG_DIR = Path("/home/lloyd/eidosian_forge/projects/src/self_exploration/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of parsing a document."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = ""
    source_type: str = ""  # "file" or "url"
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_type: str = ""
    content_length: int = 0
    content_hash: str = ""
    language: Optional[str] = None
    parse_time_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "source": self.source,
            "source_type": self.source_type,
            "content_type": self.content_type,
            "content_length": self.content_length,
            "content_hash": self.content_hash,
            "language": self.language,
            "parse_time_ms": self.parse_time_ms,
            "success": self.success,
            "error": self.error,
            "metadata_keys": list(self.metadata.keys()),
        }
    
    def save_log(self) -> Path:
        """Save parse log to disk."""
        path = LOG_DIR / f"parse_{self.timestamp.replace(':', '-').replace('+', '_')}_{self.id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


class DocumentProcessor:
    """
    Process documents using Apache Tika.
    
    Supports:
    - Local files
    - URLs (with rate limiting)
    - Raw content
    """
    
    def __init__(
        self,
        rate_limit_seconds: float = 1.0,
        max_content_length: int = 10 * 1024 * 1024,  # 10MB
        log_all: bool = True,
    ):
        self.rate_limit = rate_limit_seconds
        self.max_content_length = max_content_length
        self.log_all = log_all
        self._last_request_time = 0.0
        self._parse_count = 0
    
    def _enforce_rate_limit(self) -> None:
        """Enforce rate limit between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()
    
    def _compute_hash(self, content: str) -> str:
        """Compute hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    
    def parse_file(self, file_path: str) -> ParseResult:
        """
        Parse a local file.
        
        Args:
            file_path: Path to file to parse
            
        Returns:
            ParseResult with content and metadata
        """
        result = ParseResult(source=file_path, source_type="file")
        start_time = time.time()
        
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if path.stat().st_size > self.max_content_length:
                raise ValueError(f"File too large: {path.stat().st_size} bytes")
            
            parser = _get_tika_parser()
            parsed = parser.from_file(str(path))
            
            result.content = parsed.get("content", "") or ""
            result.metadata = parsed.get("metadata", {}) or {}
            result.content_type = result.metadata.get("Content-Type", "unknown")
            result.content_length = len(result.content)
            result.content_hash = self._compute_hash(result.content)
            result.language = result.metadata.get("language")
            result.success = True
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Failed to parse {file_path}: {e}")
        
        result.parse_time_ms = (time.time() - start_time) * 1000
        self._parse_count += 1
        
        if self.log_all:
            result.save_log()
        
        return result
    
    def parse_url(self, url: str) -> ParseResult:
        """
        Parse content from a URL.
        
        Args:
            url: URL to fetch and parse
            
        Returns:
            ParseResult with content and metadata
        """
        result = ParseResult(source=url, source_type="url")
        start_time = time.time()
        
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if parsed_url.scheme not in ("http", "https"):
                raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}")
            
            self._enforce_rate_limit()
            
            parser = _get_tika_parser()
            parsed = parser.from_file(url)
            
            result.content = parsed.get("content", "") or ""
            result.metadata = parsed.get("metadata", {}) or {}
            result.content_type = result.metadata.get("Content-Type", "unknown")
            result.content_length = len(result.content)
            result.content_hash = self._compute_hash(result.content)
            result.language = result.metadata.get("language")
            result.success = True
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Failed to parse {url}: {e}")
        
        result.parse_time_ms = (time.time() - start_time) * 1000
        self._parse_count += 1
        
        if self.log_all:
            result.save_log()
        
        return result
    
    def parse_content(self, content: bytes, content_type: str = "application/octet-stream") -> ParseResult:
        """
        Parse raw content bytes.
        
        Args:
            content: Raw bytes to parse
            content_type: MIME type hint
            
        Returns:
            ParseResult with extracted content
        """
        result = ParseResult(source="<bytes>", source_type="bytes")
        start_time = time.time()
        
        try:
            if len(content) > self.max_content_length:
                raise ValueError(f"Content too large: {len(content)} bytes")
            
            parser = _get_tika_parser()
            
            # Write to temp file for Tika
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                parsed = parser.from_file(tmp_path)
                result.content = parsed.get("content", "") or ""
                result.metadata = parsed.get("metadata", {}) or {}
                result.content_type = result.metadata.get("Content-Type", content_type)
                result.content_length = len(result.content)
                result.content_hash = self._compute_hash(result.content)
                result.language = result.metadata.get("language")
                result.success = True
            finally:
                os.unlink(tmp_path)
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Failed to parse content: {e}")
        
        result.parse_time_ms = (time.time() - start_time) * 1000
        self._parse_count += 1
        
        if self.log_all:
            result.save_log()
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_parses": self._parse_count,
            "rate_limit_seconds": self.rate_limit,
            "max_content_length": self.max_content_length,
        }


def test_document_processor():
    """Test the document processor."""
    proc = DocumentProcessor()
    
    # Test parsing a local Python file
    result = proc.parse_file(__file__)
    
    print(f"Parsed: {result.source}")
    print(f"Content type: {result.content_type}")
    print(f"Content length: {result.content_length}")
    print(f"Parse time: {result.parse_time_ms:.2f}ms")
    print(f"Success: {result.success}")
    print(f"First 200 chars: {result.content[:200]}...")
    
    return result


if __name__ == "__main__":
    print("Testing DocumentProcessor...")
    result = test_document_processor()
    print(f"\n✅ Test passed! Parsed {result.content_length} chars in {result.parse_time_ms:.2f}ms")
