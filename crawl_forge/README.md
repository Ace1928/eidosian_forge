# 🕷️ Crawl Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Eyes of Eidos (External).**

## 🕷️ Overview

`crawl_forge` provides tools for gathering information from the external web. It is designed to be ethical, safe, and structured.

## 🏗️ Architecture

- **Fetcher (`fetcher.py`)**: Handles HTTP requests with retries, user-agent rotation, and timeout management.
- **Parser (`parser.py`)**: Extracts clean text, metadata, and links using `BeautifulSoup` or `readability`.
- **Policy (`policy.py`)**: Enforces `robots.txt` compliance and rate limiting per domain.

## 🔗 System Integration

- **Eidos MCP**: Exposes `web_fetch` and `web_search` capabilities.
- **Memory Forge**: Stores crawled content for semantic indexing.

## 🚀 Usage

```python
from crawl_forge.core import CrawlForge

crawler = CrawlForge()

# Check permissions
if crawler.can_fetch("https://example.com"):
    # Fetch and parse
    result = crawler.fetch("https://example.com")
    print(result.title)
    print(result.clean_text)
```
