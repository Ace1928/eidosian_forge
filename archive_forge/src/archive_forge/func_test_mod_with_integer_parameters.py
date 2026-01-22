import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
def test_mod_with_integer_parameters(self):
    msgid = 'Some string with params: %d'
    params = [0, 1, 10, 24124]
    messages = []
    results = []
    for param in params:
        messages.append(msgid % param)
        results.append(_message.Message(msgid) % param)
    for message, result in zip(messages, results):
        self.assertIsInstance(result, _message.Message)
        self.assertEqual(message, result.translation())
        result_str = '%s' % result.translation()
        self.assertEqual(result_str, message)
        self.assertEqual(message, result)