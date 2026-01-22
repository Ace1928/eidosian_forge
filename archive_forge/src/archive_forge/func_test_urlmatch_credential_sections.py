from urllib.parse import urlparse
from dulwich.tests import TestCase
from ..config import ConfigDict
from ..credentials import match_partial_url, match_urls, urlmatch_credential_sections
def test_urlmatch_credential_sections(self):
    config = ConfigDict()
    config.set((b'credential', 'https://github.com'), b'helper', 'foo')
    config.set((b'credential', 'git.sr.ht'), b'helper', 'foo')
    config.set(b'credential', b'helper', 'bar')
    self.assertEqual(list(urlmatch_credential_sections(config, 'https://github.com')), [(b'credential', b'https://github.com'), (b'credential',)])
    self.assertEqual(list(urlmatch_credential_sections(config, 'https://git.sr.ht')), [(b'credential', b'git.sr.ht'), (b'credential',)])
    self.assertEqual(list(urlmatch_credential_sections(config, 'missing_url')), [(b'credential',)])