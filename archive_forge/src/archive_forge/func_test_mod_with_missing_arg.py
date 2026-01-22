import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
def test_mod_with_missing_arg(self):
    msgid = 'Test that we handle missing args %(arg1)s %(arg2)s'
    params = {'arg1': 'test1'}
    with testtools.ExpectedException(KeyError, '.*arg2.*'):
        _message.Message(msgid) % params