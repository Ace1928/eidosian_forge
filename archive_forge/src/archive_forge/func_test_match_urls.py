from urllib.parse import urlparse
from dulwich.tests import TestCase
from ..config import ConfigDict
from ..credentials import match_partial_url, match_urls, urlmatch_credential_sections
def test_match_urls(self):
    url = urlparse('https://github.com/jelmer/dulwich/')
    url_1 = urlparse('https://github.com/jelmer/dulwich')
    url_2 = urlparse('https://github.com/jelmer')
    url_3 = urlparse('https://github.com')
    self.assertTrue(match_urls(url, url_1))
    self.assertTrue(match_urls(url, url_2))
    self.assertTrue(match_urls(url, url_3))
    non_matching = urlparse('https://git.sr.ht/')
    self.assertFalse(match_urls(url, non_matching))