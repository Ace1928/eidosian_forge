# 🕸️ Crawl Forge ⚡

```ascii
     ______ ____  ___ __      __ _       __________  ____  ____________
    / ____// __ \/   |\ \    / // /      / ____/ __ \/ __ \/ ____/ ____/
   / /    / /_/ / /| | \ \  / // /      / /_  / / / / /_/ / / __/ __/   
  / /___ / _, _/ ___ |  \ \/ // /___   / __/ / /_/ / _, _/ /_/ / /___   
  \____//_/ |_/_/  |_|   \__/ /_____/  /_/    \____/_/ |_|\____/_____/   
                                                                         
```

> _"Intelligent and ethical data harvesting. Expanding the Eidosian knowledge perimeter with respect for boundaries."_

## 🧠 Overview

`crawl_forge` is the ingestion engine responsible for ethical web fetching, structured data extraction, and deep document parsing. It bridges the gap between unstructured external data and the Eidosian Knowledge/Memory substrates. Designed for performance in restricted environments (Termux), it strictly enforces crawler politeness (`robots.txt`) and utilizes multi-layered caching to minimize systemic footprint.

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Data Ingestion & Harvesting
- **Test Coverage**: 100% Core Logic validated.
- **Inference Standard**: **Qwen 3.5 2B** (Used for semantic content triage).
- **Core Architecture**:
  - **`CrawlForge`**: High-level orchestrator for ethical fetching and metadata extraction.
  - **`TikaExtractor`**: Multi-format deep inspection (PDF, Office, Images/OCR) with adaptive fallback.
  - **`PolitenessRegistry`**: Cache-aware management of `robots.txt` and rate-limiting policies.
  - **`TikaKnowledgeIngester`**: Direct actuator for feeding extracted semantic units into `knowledge_forge`.

## 🛡️ Ethics & Compliance

1. **Strict Robot Policy**: Every URL is verified against `robots.txt` before fetching.
2. **User-Agent Transparency**: Operates with a clearly defined Eidosian User-Agent header.
3. **Adaptive Rate Limiting**: Enforces cooldowns based on domain hit frequency to prevent systemic pressure.
4. **Cache First**: Prioritizes local JSON-serialized snapshots to avoid redundant external requests.

## 🚀 Usage & Workflows

### Deep Document Extraction (Tika)

Extract structured text and metadata from a local or remote PDF:
```bash
python -m crawl_forge.cli tika /path/to/research_paper.pdf
```

### Ethical Web Extraction

Safely fetch and extract basic metadata (title, links, summary):
```bash
python -m crawl_forge.cli extract https://example.com/blog/post
```

### Python API (Autonomous Ingestion)

```python
from crawl_forge import CrawlForge

crawler = CrawlForge(enable_http_cache=True)

if crawler.can_fetch("https://docs.eidos.io"):
    content = crawler.fetch_page("https://docs.eidos.io")
    data = crawler.extract_structured_data(content)
    print(f"Discovered {len(data['links'])} new semantic paths.")
```

## 🔗 System Integration

- **Knowledge Forge**: Feeds the "Ingestion" missions with fresh nodes derived from document scans.
- **Eidos MCP**: Exposes 8+ specialized tools for ad-hoc web research and document parsing.
- **Word Forge**: Extracted text is tokenized and analyzed for emotional valence during ingestion.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy extraction documentation.
- [x] Implement adaptive SSL fallback for Termux/Android environments.
- [x] Integrate Eidosian observability decorators across the Tika pipeline.

### Future Vector (Phase 3+)
- Add headless browser support (Playwright) for JS-heavy, client-side rendered applications.
- Build "Semantic Chunking" logic that groups extracted text by concept density before Knowledge Forge injection.
- Implement "Continuous Monitoring" for specific technical documentation hubs to auto-detect updates.

---
*Generated and maintained by Eidos.*
