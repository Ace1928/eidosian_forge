# Crawl Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Eyes of Eidos (External).**

## 🕷️ Overview

`crawl_forge` provides tools for gathering information from the external web.
It prioritizes:
- **Ethics**: Respects `robots.txt` strictly.
- **Safety**: Rate limiting and timeout management.
- **Structure**: Extracts clean data (Title, Metadata) for the Knowledge Graph.

## 🏗️ Architecture
- `crawl_core.py`: Main `CrawlForge` class.

## 🚀 Usage

```python
from crawl_forge.crawl_core import CrawlForge

crawler = CrawlForge()
if crawler.can_fetch("https://example.com"):
    html = crawler.fetch_page("https://example.com")
    data = crawler.extract_structured_data(html)
    print(data)
```