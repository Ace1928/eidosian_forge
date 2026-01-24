# Crawl Forge Installation Guide

## Quick Install

```bash
# From the crawl_forge directory
pip install -e .

# Or install with Tika support
pip install -e ".[tika]"
```

## Prerequisites

For full Tika integration, install Java and Tika:
```bash
# Java (required for Tika)
sudo apt install default-jdk

# Tika will auto-download on first use
```

## Verify Installation

```bash
# Check CLI is available
crawl-forge --version

# Check system status
crawl-forge status
```

## Enable Bash Completions

```bash
# Add to your ~/.bashrc
source /home/lloyd/eidosian_forge/crawl_forge/completions/crawl-forge.bash
```

## Standalone vs. Integrated Usage

### Standalone
Crawl Forge works independently with these capabilities:
- Ethical web crawling with robots.txt compliance
- Rate limiting
- Basic HTML extraction
- Tika-powered deep extraction

### Enhanced with Other Forges
When other forges are installed, additional capabilities are enabled:

| Forge | Enhanced Capability |
|-------|---------------------|
| `knowledge_forge` | Auto-ingest extracted content |

## CLI Commands

```bash
crawl-forge status              # Show crawler status
crawl-forge fetch <url>         # Fetch a URL
crawl-forge extract <url>       # Extract structured content
crawl-forge robots <url>        # Check robots.txt
crawl-forge tika <path>         # Extract with Tika
crawl-forge cache               # Show Tika cache status
```

## Python API

```python
from crawl_forge import CrawlForge, TikaExtractor

# Basic crawling
crawler = CrawlForge()
content = crawler.fetch_page("https://example.com")
data = crawler.extract_structured_data(content)

# Check robots.txt
if crawler.can_fetch("https://example.com/page"):
    content = crawler.fetch_page("https://example.com/page")

# Tika extraction
tika = TikaExtractor()
result = tika.extract_from_url("https://example.com/document.pdf")
print(result["text"])
```

## Troubleshooting

### Tika Server Not Running
Tika auto-starts on first use. If issues persist:
```bash
# Check Java installation
java -version
```

### Rate Limiting
The crawler respects rate limits (1 request/second by default).
