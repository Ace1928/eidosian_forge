# 🕸️ Crawl Forge ⚡

> _"Intelligent and ethical data harvesting. Expanding the Eidosian knowledge perimeter with respect for boundaries."_

## 🧠 Overview

`crawl_forge` provides ethical web fetching, structured data extraction, and deep document parsing (via Apache Tika). It is designed with cache-aware behavior optimized for Termux/Linux runs, strictly enforcing crawler politeness (`robots.txt` compliance) before any external interaction.

```ascii
      ╭───────────────────────────────────────────╮
      │               CRAWL FORGE                 │
      │    < Fetch | Extract | Tika Deep Scan >   │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   HTTP/ROBOT CACHE  │   │  KNOWLEDGE INJ. │
      │ (Rate Limited)      │   │ (Graph/Memory)  │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Data Ingestion & Harvesting
- **Test Coverage**: 100% Core Logic Validated
- **Core Capabilities**:
  - `CrawlForge`: Basic fetching, HTML metadata extraction (BeautifulSoup), strict `robots.txt` enforcement.
  - `TikaExtractor`: Deep extraction for PDFs, Office docs, Images (OCR), and complex HTML.
  - `TikaKnowledgeIngester`: Direct pipeline from document to `KnowledgeForge` nodes.

## 🚀 Usage & Workflows

### CLI Interface

```bash
# Check system and cache health
python -m crawl_forge.cli status

# Safely fetch and extract basic metadata from a URL
python -m crawl_forge.cli extract https://example.com

# Deep extraction using Apache Tika (if available)
python -m crawl_forge.cli tika /path/to/document.pdf
python -m crawl_forge.cli tika https://example.com/report.pdf
```

### Python API

```python
from crawl_forge import CrawlForge

crawler = CrawlForge(
    enable_http_cache=True,
    http_cache_ttl_seconds=120.0,
    robots_cache_ttl_seconds=1800.0,
)

if crawler.can_fetch("https://example.com"):
    html = crawler.fetch_page("https://example.com")
    if html:
        data = crawler.extract_structured_data(html)
        print(f"Title: {data['title']}, Links: {len(data['links'])}")
```

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into a unified Eidosian standard.
- [x] Deep integration with Tika extraction pipeline.
- [x] Integrate directly into MCP ecosystem.

### Future Vector (Phase 3+)
- Add headless browser support (Playwright/Selenium) for JS-heavy client-side rendered applications.
- Build advanced semantic chunking for raw text prior to Knowledge Forge ingestion.

---
*Generated and maintained by Eidos.*
