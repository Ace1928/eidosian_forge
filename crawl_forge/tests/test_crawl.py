import unittest
from unittest.mock import MagicMock, patch
from eidosian_forge.crawl_forge import CrawlForge

class TestCrawlForge(unittest.TestCase):
    def setUp(self):
        self.forge = CrawlForge()

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

if __name__ == "__main__":
    unittest.main()
