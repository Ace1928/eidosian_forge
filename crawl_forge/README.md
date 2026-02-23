# Crawl Forge

`crawl_forge` provides ethical web fetching and structured extraction with cache-aware behavior for Termux/Linux runs.

## Capabilities

- `CrawlForge.can_fetch(url)`: checks `robots.txt` policy.
- `CrawlForge.fetch_page(url)`: rate-limited fetch with HTTP response cache.
- `CrawlForge.extract_structured_data(html)`: HTML metadata extraction via BeautifulSoup.
- `CrawlForge.cache_stats()`: runtime cache visibility for page and robots caches.
- `TikaExtractor` / `TikaKnowledgeIngester`: optional deep document extraction and ingestion (when `tika` extras are installed).

## Runtime Caching

- `robots.txt` parser cache with TTL (`robots_cache_ttl_seconds`).
- HTTP content cache with TTL (`http_cache_ttl_seconds`).
- Both caches are in-memory and safe to disable (`enable_http_cache=False`).

## CLI

```bash
crawl-forge status
crawl-forge fetch https://example.com
crawl-forge extract https://example.com
crawl-forge robots https://example.com
crawl-forge tika /path/to/document.pdf
```

`crawl-forge status` reports cache and Tika availability in addition to crawler settings.

## Python Usage

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
        print(data["title"], len(data["links"]))
```

## Notes

- This forge intentionally enforces crawler politeness before fetching.
- For JS-heavy pages, headless browser support remains a planned TODO item.
