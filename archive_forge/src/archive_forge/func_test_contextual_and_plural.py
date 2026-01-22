import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
@mock.patch('gettext.translation')
def test_contextual_and_plural(self, translation):
    self.assertRaises(ValueError, _message.Message._translate_msgid, 'nothing', domain='domain', has_contextual_form=True, has_plural_form=True)