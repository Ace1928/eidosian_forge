import builtins
import gettext
import logging
from unittest import mock
from oslotest import base as test_base
from oslo_i18n import _factory
from oslo_i18n import _gettextutils
from oslo_i18n import _lazy
from oslo_i18n import _message
def test_cached_find(self):
    domain = 'my-unique-domain'
    key = (domain, None, None, 0)
    self.assertNotIn(key, _gettextutils._FIND_CACHE)
    gettext.find(domain)
    self.assertIn(key, _gettextutils._FIND_CACHE)
    _gettextutils._FIND_CACHE[key] = 'spoof result'
    self.assertEqual('spoof result', gettext.find(domain))
    _gettextutils._FIND_CACHE.pop(key)