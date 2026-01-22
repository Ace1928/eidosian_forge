import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
def test_mod_deep_copies_param_nodeep_param(self):
    msgid = 'Value: %s'
    params = utils.NoDeepCopyObject(5)
    result = _message.Message(msgid) % params
    self.assertEqual('Value: 5', result.translation())