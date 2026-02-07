"""
Tika integration for document extraction.

Apache Tika can extract text and metadata from:
- PDF documents
- Microsoft Office files (Word, Excel, PowerPoint)
- HTML pages
- Plain text
- Images (with OCR)
- Many other formats
"""

from __future__ import annotations
from eidosian_core import eidosian

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_TIKA_AVAILABLE: bool | None = None
_TIKA_PARSER = None


def _get_tika_parser() -> Optional[Any]:
    global _TIKA_AVAILABLE, _TIKA_PARSER
    if _TIKA_AVAILABLE is False:
        return None
    if _TIKA_PARSER is None:
        try:
            from tika import parser as tika_parser
        except ImportError:
            _TIKA_AVAILABLE = False
            return None
        _TIKA_PARSER = tika_parser
        _TIKA_AVAILABLE = True
    return _TIKA_PARSER


class TikaExtractor:
    """
    Document extraction using Apache Tika.
    
    Supports both local files and URLs.
    Caches extracted content locally for performance.
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        enable_cache: bool = True,
    ):
        """
        Initialize the Tika extractor.
        
        Args:
            cache_dir: Directory to cache extracted content
            enable_cache: Whether to use caching
        """
        parser = _get_tika_parser()
        if not parser:
            raise ImportError(
                "Apache Tika not available. Install with: pip install tika"
            )

        self._parser = parser
        
        self.enable_cache = enable_cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".eidosian" / "tika_cache"
        
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _cache_key(self, source: str) -> str:
        """Generate a cache key from a source URL or path."""
        return hashlib.sha256(source.encode()).hexdigest()[:32]
    
    def _get_cached(self, source: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached extraction result."""
        if not self.enable_cache:
            return None
        
        cache_file = self.cache_dir / f"{self._cache_key(source)}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Cache read failed for {source}: {e}")
        return None
    
    def _set_cached(self, source: str, data: Dict[str, Any]) -> None:
        """Store extraction result in cache."""
        if not self.enable_cache:
            return
        
        cache_file = self.cache_dir / f"{self._cache_key(source)}.json"
        try:
            cache_file.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Cache write failed for {source}: {e}")
    
    @eidosian()
    def extract_from_file(
        self,
        file_path: Path,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract text and metadata from a local file.
        
        Args:
            file_path: Path to the file
            use_cache: Whether to use/update cache for this extraction
            
        Returns:
            Dict with 'content', 'metadata', 'status', 'source'
        """
        file_path = Path(file_path)
        source = str(file_path.absolute())
        
        # Check cache
        if use_cache:
            cached = self._get_cached(source)
            if cached:
                cached["from_cache"] = True
                return cached
        
        if not file_path.exists():
            return {
                "content": None,
                "metadata": {},
                "status": "error",
                "error": f"File not found: {file_path}",
                "source": source,
            }
        
        try:
            result = self._parser.from_file(str(file_path))
            extracted = {
                "content": result.get("content", ""),
                "metadata": result.get("metadata", {}),
                "status": "success" if result.get("content") else "empty",
                "source": source,
                "from_cache": False,
            }
            
            # Cache the result
            if use_cache:
                self._set_cached(source, extracted)
            
            return extracted
            
        except Exception as e:
            logger.error(f"Tika extraction failed for {file_path}: {e}")
            return {
                "content": None,
                "metadata": {},
                "status": "error",
                "error": str(e),
                "source": source,
            }
    
    @eidosian()
    def extract_from_url(
        self,
        url: str,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract text and metadata from a URL.
        
        Args:
            url: URL to extract from
            use_cache: Whether to use/update cache
            
        Returns:
            Dict with 'content', 'metadata', 'status', 'source'
        """
        # Check cache
        if use_cache:
            cached = self._get_cached(url)
            if cached:
                cached["from_cache"] = True
                return cached
        
        try:
            result = self._parser.from_url(url)
            extracted = {
                "content": result.get("content", ""),
                "metadata": result.get("metadata", {}),
                "status": "success" if result.get("content") else "empty",
                "source": url,
                "from_cache": False,
            }
            
            # Cache the result
            if use_cache:
                self._set_cached(url, extracted)
            
            return extracted
            
        except Exception as e:
            logger.error(f"Tika extraction failed for {url}: {e}")
            return {
                "content": None,
                "metadata": {},
                "status": "error",
                "error": str(e),
                "source": url,
            }
    
    @eidosian()
    def extract_from_buffer(
        self,
        buffer: bytes,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract text and metadata from a byte buffer.
        
        Args:
            buffer: Raw bytes of the document
            filename: Optional filename hint for content type detection
            
        Returns:
            Dict with 'content', 'metadata', 'status'
        """
        try:
            result = self._parser.from_buffer(buffer)
            return {
                "content": result.get("content", ""),
                "metadata": result.get("metadata", {}),
                "status": "success" if result.get("content") else "empty",
                "source": filename or "buffer",
            }
        except Exception as e:
            logger.error(f"Tika buffer extraction failed: {e}")
            return {
                "content": None,
                "metadata": {},
                "status": "error",
                "error": str(e),
                "source": filename or "buffer",
            }
    
    @eidosian()
    def get_metadata_only(
        self,
        file_path: Path,
    ) -> Dict[str, Any]:
        """
        Extract only metadata (faster than full extraction).
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict with metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        try:
            # Use from_file with requestOptions to get only metadata
            result = self._parser.from_file(
                str(file_path),
                requestOptions={"headers": {"Accept": "application/json"}}
            )
            return result.get("metadata", {})
        except Exception as e:
            logger.error(f"Tika metadata extraction failed for {file_path}: {e}")
            return {"error": str(e)}
    
    @eidosian()
    def clear_cache(self, source: Optional[str] = None) -> int:
        """
        Clear cached extractions.
        
        Args:
            source: Specific source to clear, or None to clear all
            
        Returns:
            Number of cache entries cleared
        """
        if not self.enable_cache:
            return 0
        
        if source:
            cache_file = self.cache_dir / f"{self._cache_key(source)}.json"
            if cache_file.exists():
                cache_file.unlink()
                return 1
            return 0
        
        # Clear all
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count
    
    @eidosian()
    def cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        if not self.enable_cache:
            return {"enabled": False}
        
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "entries": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }


class TikaKnowledgeIngester:
    """
    Bridge between Tika extraction and Knowledge Forge.
    
    Extracts content from documents and ingests them into the knowledge graph.
    """
    
    def __init__(
        self,
        tika: TikaExtractor,
        knowledge_forge: Any = None,
    ):
        """
        Initialize the knowledge ingester.
        
        Args:
            tika: TikaExtractor instance
            knowledge_forge: KnowledgeForge instance for ingestion
        """
        self.tika = tika
        self.knowledge_forge = knowledge_forge
    
    @eidosian()
    def ingest_file(
        self,
        file_path: Path,
        tags: Optional[list] = None,
        chunk_size: int = 2000,
    ) -> Dict[str, Any]:
        """
        Extract and ingest a file into knowledge forge.
        
        Args:
            file_path: Path to the file
            tags: Tags to apply to knowledge nodes
            chunk_size: Maximum characters per knowledge chunk
            
        Returns:
            Dict with ingestion results
        """
        file_path = Path(file_path)
        tags = tags or []
        
        # Extract content
        extracted = self.tika.extract_from_file(file_path)
        
        if extracted["status"] == "error":
            return {
                "status": "error",
                "error": extracted.get("error"),
                "source": str(file_path),
            }
        
        content = extracted.get("content", "")
        if not content or not content.strip():
            return {
                "status": "empty",
                "source": str(file_path),
                "nodes_created": 0,
            }
        
        # Add file-specific tags
        file_tags = list(tags) + [
            f"source:{file_path.name}",
            f"type:{file_path.suffix.lstrip('.')}" if file_path.suffix else "type:unknown",
        ]
        
        # Chunk and ingest
        nodes_created = 0
        if self.knowledge_forge:
            chunks = self._chunk_content(content, chunk_size)
            for i, chunk in enumerate(chunks):
                chunk_tags = file_tags + [f"chunk:{i+1}/{len(chunks)}"]
                try:
                    self.knowledge_forge.add_knowledge(chunk.strip(), tags=chunk_tags)
                    nodes_created += 1
                except Exception as e:
                    logger.warning(f"Failed to add chunk {i+1}: {e}")
        
        return {
            "status": "success",
            "source": str(file_path),
            "content_length": len(content),
            "chunks": nodes_created if self.knowledge_forge else 0,
            "nodes_created": nodes_created,
            "metadata": extracted.get("metadata", {}),
        }
    
    @eidosian()
    def ingest_url(
        self,
        url: str,
        tags: Optional[list] = None,
        chunk_size: int = 2000,
    ) -> Dict[str, Any]:
        """
        Extract and ingest a URL into knowledge forge.
        
        Args:
            url: URL to ingest
            tags: Tags to apply to knowledge nodes
            chunk_size: Maximum characters per knowledge chunk
            
        Returns:
            Dict with ingestion results
        """
        tags = tags or []
        
        # Extract content
        extracted = self.tika.extract_from_url(url)
        
        if extracted["status"] == "error":
            return {
                "status": "error",
                "error": extracted.get("error"),
                "source": url,
            }
        
        content = extracted.get("content", "")
        if not content or not content.strip():
            return {
                "status": "empty",
                "source": url,
                "nodes_created": 0,
            }
        
        # Parse URL for tags
        parsed = urlparse(url)
        url_tags = list(tags) + [
            f"source:web",
            f"domain:{parsed.netloc}",
        ]
        
        # Chunk and ingest
        nodes_created = 0
        if self.knowledge_forge:
            chunks = self._chunk_content(content, chunk_size)
            for i, chunk in enumerate(chunks):
                chunk_tags = url_tags + [f"chunk:{i+1}/{len(chunks)}"]
                try:
                    self.knowledge_forge.add_knowledge(chunk.strip(), tags=chunk_tags)
                    nodes_created += 1
                except Exception as e:
                    logger.warning(f"Failed to add chunk {i+1}: {e}")
        
        return {
            "status": "success",
            "source": url,
            "content_length": len(content),
            "chunks": nodes_created if self.knowledge_forge else 0,
            "nodes_created": nodes_created,
            "metadata": extracted.get("metadata", {}),
            "from_cache": extracted.get("from_cache", False),
        }
    
    def _chunk_content(self, content: str, chunk_size: int) -> list:
        """Split content into chunks, trying to break at paragraph boundaries."""
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        paragraphs = content.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If a single paragraph is too long, split it
                if len(para) > chunk_size:
                    words = para.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= chunk_size:
                            current_chunk += word + " "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = word + " "
                else:
                    current_chunk = para + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
