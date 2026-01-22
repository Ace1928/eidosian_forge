import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
def test_message_id_and_message_text(self):
    message = _message.Message('1')
    self.assertEqual('1', message.msgid)
    self.assertEqual('1', message)
    message = _message.Message('1', msgtext='A')
    self.assertEqual('1', message.msgid)
    self.assertEqual('A', message)