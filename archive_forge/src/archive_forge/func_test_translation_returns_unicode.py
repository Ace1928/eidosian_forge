import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
def test_translation_returns_unicode(self):
    message = _message.Message('some %s') % 'message'
    self.assertIsInstance(message.translation(), str)