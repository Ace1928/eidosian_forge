import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
def test_mod_returns_a_copy(self):
    msgid = 'Some msgid string: %(test1)s %(test2)s'
    message = _message.Message(msgid)
    m1 = message % {'test1': 'foo', 'test2': 'bar'}
    m2 = message % {'test1': 'foo2', 'test2': 'bar2'}
    self.assertIsNot(message, m1)
    self.assertIsNot(message, m2)
    self.assertEqual(m1.translation(), msgid % {'test1': 'foo', 'test2': 'bar'})
    self.assertEqual(m2.translation(), msgid % {'test1': 'foo2', 'test2': 'bar2'})