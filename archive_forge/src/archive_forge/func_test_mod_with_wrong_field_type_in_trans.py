import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
def test_mod_with_wrong_field_type_in_trans(self):
    msgid = 'Correct type %(arg1)s'
    params = {'arg1': 'test1'}
    with mock.patch('gettext.translation') as trans:
        trans.return_value.ugettext.return_value = msgid
        result = _message.Message(msgid) % params
        wrong_type = 'Wrong type %(arg1)d'
        trans.return_value.gettext.return_value = wrong_type
        trans_result = result.translation()
        expected = msgid % params
        self.assertEqual(expected, trans_result)