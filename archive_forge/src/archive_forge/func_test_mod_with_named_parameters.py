import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
def test_mod_with_named_parameters(self):
    msgid = '%(description)s\nCommand: %(cmd)s\nExit code: %(exit_code)s\nStdout: %(stdout)r\nStderr: %(stderr)r %%(something)s'
    params = {'description': 'test1', 'cmd': 'test2', 'exit_code': 'test3', 'stdout': 'test4', 'stderr': 'test5', 'something': 'trimmed'}
    result = _message.Message(msgid) % params
    expected = msgid % params
    self.assertEqual(expected, result)
    self.assertEqual(expected, result.translation())