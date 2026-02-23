#!/usr/bin/env python3
"""
Crawl Forge CLI - Command-line interface for web crawling and extraction.

Standalone Usage:
    crawl-forge status              # Show crawler status
    crawl-forge fetch <url>         # Fetch a URL
    crawl-forge extract <url>       # Extract content from URL
    crawl-forge robots <url>        # Check robots.txt

Enhanced with other forges:
    - knowledge_forge: Auto-ingest extracted content
    - tika: Deep content extraction
"""
from __future__ import annotations
from eidosian_core import eidosian

import sys
from pathlib import Path
from typing import Optional

# Add lib to path for CLI framework
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "lib"))

from cli import StandardCLI, CommandResult, ForgeDetector

from crawl_forge import CrawlForge


class CrawlForgeCLI(StandardCLI):
    """CLI for Crawl Forge - web crawling and extraction."""
    
    name = "crawl_forge"
    description = "Ethical web crawling with content extraction and Tika integration"
    version = "1.0.0"
    
    def __init__(self):
        super().__init__()
        self._crawler: Optional[CrawlForge] = None
        self._tika: Optional[object] = None
    
    @property
    def crawler(self) -> CrawlForge:
        """Lazy-load crawler."""
        if self._crawler is None:
            self._crawler = CrawlForge()
        return self._crawler
    
    @property
    def tika(self):
        """Lazy-load Tika extractor if available."""
        if self._tika is None:
            try:
                from crawl_forge import TikaExtractor
                if TikaExtractor:
                    self._tika = TikaExtractor()
            except Exception:
                pass
        return self._tika
    
    @eidosian()
    def register_commands(self, subparsers) -> None:
        """Register crawl-forge specific commands."""
        
        # Fetch command
        fetch_parser = subparsers.add_parser(
            "fetch",
            help="Fetch a URL",
        )
        fetch_parser.add_argument(
            "url",
            help="URL to fetch",
        )
        fetch_parser.add_argument(
            "-o", "--output",
            help="Save to file",
        )
        fetch_parser.set_defaults(func=self._cmd_fetch)
        
        # Extract command
        extract_parser = subparsers.add_parser(
            "extract",
            help="Extract structured content from URL",
        )
        extract_parser.add_argument(
            "url",
            help="URL to extract from",
        )
        extract_parser.add_argument(
            "--tika",
            action="store_true",
            help="Use Tika for deep extraction",
        )
        extract_parser.set_defaults(func=self._cmd_extract)
        
        # Robots command
        robots_parser = subparsers.add_parser(
            "robots",
            help="Check robots.txt for URL",
        )
        robots_parser.add_argument(
            "url",
            help="URL to check",
        )
        robots_parser.set_defaults(func=self._cmd_robots)
        
        # Tika command
        tika_parser = subparsers.add_parser(
            "tika",
            help="Extract content using Tika",
        )
        tika_parser.add_argument(
            "path",
            help="File path or URL",
        )
        tika_parser.set_defaults(func=self._cmd_tika)
        
        # Cache command
        cache_parser = subparsers.add_parser(
            "cache",
            help="Show Tika cache status",
        )
        cache_parser.set_defaults(func=self._cmd_cache)
    
    @eidosian()
    def cmd_status(self, args) -> CommandResult:
        """Show crawl forge status."""
        try:
            # Check components
            components = {
                "crawler": True,
                "tika": self.tika is not None,
            }
            
            # Check Tika server
            tika_server = False
            if self.tika:
                try:
                    import requests
                    resp = requests.get("http://localhost:9998/tika", timeout=2)
                    tika_server = resp.status_code == 200
                except Exception:
                    pass
            
            integrations = []
            if ForgeDetector.is_available("knowledge_forge"):
                integrations.append("knowledge_forge")
            
            return CommandResult(
                True,
                f"Crawl Forge operational - Tika: {'connected' if tika_server else 'not connected'}",
                {
                    "components": components,
                    "tika_server": tika_server,
                    "user_agent": self.crawler.user_agent,
                    "rate_limit": self.crawler.rate_limit,
                    "cache": self.crawler.cache_stats(),
                    "integrations": integrations,
                }
            )
        except Exception as e:
            return CommandResult(False, f"Error: {e}")
    
    def _cmd_fetch(self, args) -> None:
        """Fetch a URL."""
        try:
            content = self.crawler.fetch_page(args.url)
            
            if content is None:
                result = CommandResult(
                    False,
                    f"Could not fetch {args.url}",
                    {"url": args.url, "success": False}
                )
            else:
                if args.output:
                    Path(args.output).write_text(content)
                    result = CommandResult(
                        True,
                        f"Saved {len(content)} bytes to {args.output}",
                        {"url": args.url, "size": len(content), "output": args.output}
                    )
                else:
                    result = CommandResult(
                        True,
                        f"Fetched {len(content)} bytes from {args.url}",
                        {"url": args.url, "size": len(content), "preview": content[:200]}
                    )
        except Exception as e:
            result = CommandResult(False, f"Fetch error: {e}")
        self._output(result, args)
    
    def _cmd_extract(self, args) -> None:
        """Extract structured content."""
        try:
            content = self.crawler.fetch_page(args.url)
            
            if content is None:
                result = CommandResult(
                    False,
                    f"Could not fetch {args.url}",
                    {"url": args.url}
                )
            else:
                if args.tika and self.tika:
                    # Use Tika for extraction
                    extracted = self.tika.extract_from_url(args.url)
                    result = CommandResult(
                        True,
                        f"Extracted with Tika from {args.url}",
                        {
                            "url": args.url,
                            "method": "tika",
                            "content_type": extracted.get("content_type", "unknown"),
                            "text_length": len(extracted.get("text", "")),
                        }
                    )
                else:
                    # Use basic extraction
                    data = self.crawler.extract_structured_data(content)
                    result = CommandResult(
                        True,
                        f"Extracted from {args.url}: {data.get('title', 'No title')}",
                        {
                            "url": args.url,
                            "method": "basic",
                            "title": data.get("title"),
                            "description": data.get("meta_description"),
                            "link_count": len(data.get("links", [])),
                        }
                    )
        except Exception as e:
            result = CommandResult(False, f"Extract error: {e}")
        self._output(result, args)
    
    def _cmd_robots(self, args) -> None:
        """Check robots.txt."""
        try:
            can_fetch = self.crawler.can_fetch(args.url)
            result = CommandResult(
                True,
                f"{'Allowed' if can_fetch else 'Disallowed'} to fetch {args.url}",
                {"url": args.url, "allowed": can_fetch}
            )
        except Exception as e:
            result = CommandResult(False, f"Robots check error: {e}")
        self._output(result, args)
    
    def _cmd_tika(self, args) -> None:
        """Extract content using Tika."""
        if not self.tika:
            result = CommandResult(False, "Tika extractor not available")
        else:
            try:
                path = Path(args.path)
                if path.exists():
                    # Local file
                    extracted = self.tika.extract_from_file(path)
                else:
                    # URL
                    extracted = self.tika.extract_from_url(args.path)
                
                result = CommandResult(
                    True,
                    f"Extracted: {extracted.get('content_type', 'unknown')}",
                    {
                        "path": args.path,
                        "content_type": extracted.get("content_type"),
                        "text_length": len(extracted.get("text", "")),
                        "metadata_keys": list(extracted.get("metadata", {}).keys())[:10],
                    }
                )
            except Exception as e:
                result = CommandResult(False, f"Tika error: {e}")
        self._output(result, args)
    
    def _cmd_cache(self, args) -> None:
        """Show Tika cache status."""
        if not self.tika:
            result = CommandResult(False, "Tika extractor not available")
        else:
            try:
                cache_dir = self.tika.cache_dir if hasattr(self.tika, 'cache_dir') else None
                
                if cache_dir and cache_dir.exists():
                    cache_files = list(cache_dir.glob("*.json"))
                    result = CommandResult(
                        True,
                        f"Cache: {len(cache_files)} items",
                        {
                            "cache_dir": str(cache_dir),
                            "items": len(cache_files),
                        }
                    )
                else:
                    result = CommandResult(
                        True,
                        "Cache empty or not configured",
                        {"cache_dir": str(cache_dir) if cache_dir else None}
                    )
            except Exception as e:
                result = CommandResult(False, f"Cache error: {e}")
        self._output(result, args)


@eidosian()
def main():
    """Entry point for crawl-forge CLI."""
    cli = CrawlForgeCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
