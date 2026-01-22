import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_apply_multiple(self):
    config = ConfigDict()
    config.set(('url', 'https://samba.org/'), 'insteadOf', 'https://blah.com/')
    config.set(('url', 'https://samba.org/'), 'insteadOf', 'https://example.com/')
    self.assertEqual([b'https://blah.com/', b'https://example.com/'], list(config.get_multivar(('url', 'https://samba.org/'), 'insteadOf')))
    self.assertEqual('https://samba.org/', apply_instead_of(config, 'https://example.com/'))