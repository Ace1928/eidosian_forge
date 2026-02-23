import unittest
from unittest.mock import MagicMock, patch
from crawl_forge import CrawlForge

class TestCrawlForge(unittest.TestCase):
    def setUp(self):
        self.forge = CrawlForge(http_cache_ttl_seconds=60.0, robots_cache_ttl_seconds=60.0)
        self.forge.rate_limit = 0.0

    @patch('requests.get')
    def test_fetch_page_allowed(self, mock_get):
        # Mock robots.txt to allow everything
        self.forge.can_fetch = MagicMock(return_value=True)
        
        mock_response = MagicMock()
        mock_response.text = "<html>Content</html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        content = self.forge.fetch_page("https://example.com")
        self.assertEqual(content, "<html>Content</html>")
        self.assertEqual(mock_get.call_count, 1)

    @patch('requests.get')
    def test_fetch_page_uses_cache(self, mock_get):
        self.forge.can_fetch = MagicMock(return_value=True)

        resp1 = MagicMock()
        resp1.text = "<html>Cached</html>"
        resp1.raise_for_status = MagicMock()
        mock_get.return_value = resp1

        first = self.forge.fetch_page("https://cached.example")
        second = self.forge.fetch_page("https://cached.example")
        self.assertEqual(first, "<html>Cached</html>")
        self.assertEqual(second, "<html>Cached</html>")
        self.assertEqual(mock_get.call_count, 1)

    @patch('requests.get')
    def test_fetch_page_cache_expiry_refetches(self, mock_get):
        self.forge.can_fetch = MagicMock(return_value=True)
        self.forge.http_cache_ttl_seconds = 1.0

        resp1 = MagicMock()
        resp1.text = "<html>V1</html>"
        resp1.raise_for_status = MagicMock()
        resp2 = MagicMock()
        resp2.text = "<html>V2</html>"
        resp2.raise_for_status = MagicMock()
        mock_get.side_effect = [resp1, resp2]

        url = "https://expiring.example"
        first = self.forge.fetch_page(url)
        self.assertEqual(first, "<html>V1</html>")
        self.forge._page_cache[url]["fetched_at"] = self.forge._page_cache[url]["fetched_at"] - 999.0
        second = self.forge.fetch_page(url)
        self.assertEqual(second, "<html>V2</html>")
        self.assertEqual(mock_get.call_count, 2)

    def test_extraction(self):
        html = """
        <html>
            <head>
                <title>Test Page</title>
                <meta name="description" content="A page for testing">
            </head>
            <body>
                <a href="https://link1.com">Link 1</a>
                <a href="https://link2.com">Link 2</a>
            </body>
        </html>
        """
        data = self.forge.extract_structured_data(html)
        self.assertEqual(data["title"], "Test Page")
        self.assertEqual(data["meta_description"], "A page for testing")
        self.assertIn("https://link1.com", data["links"])
        self.assertEqual(len(data["links"]), 2)

    def test_extraction_uses_meta_fallback_and_filters_links(self):
        html = """
        <html>
            <head>
                <title>Another Page</title>
                <meta property="og:description" content="Open graph description">
            </head>
            <body>
                <a href="#anchor">Anchor</a>
                <a href="/relative/path">Relative</a>
                <a href="https://a.example.com">A</a>
                <a href="https://a.example.com">A duplicate</a>
            </body>
        </html>
        """
        data = self.forge.extract_structured_data(html)
        self.assertEqual(data["title"], "Another Page")
        self.assertEqual(data["meta_description"], "Open graph description")
        self.assertEqual(data["links"], ["https://a.example.com"])

    def test_cache_stats(self):
        stats = self.forge.cache_stats()
        self.assertTrue(stats["http_cache_enabled"])
        self.assertIn("http_cache_items", stats)
        self.assertIn("robots_cache_items", stats)

if __name__ == "__main__":
    unittest.main()
