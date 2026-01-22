import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
@mock.patch('gettext.translation')
def test_translate_message_with_param_from_unicoded_obj(self, mock_translation):
    message_with_params = 'A message: %s'
    es_translation = 'A message in Spanish: %s'
    param = 'A Message param'
    translations = {message_with_params: es_translation}
    translator = fakes.FakeTranslations.translator({'es': translations})
    mock_translation.side_effect = translator
    msg = _message.Message(message_with_params)
    msg = msg % param
    default_translation = message_with_params % param
    expected_translation = es_translation % param
    obj = utils.SomeObject(msg)
    unicoded_obj = str(obj)
    self.assertEqual(expected_translation, unicoded_obj.translation('es'))
    self.assertEqual(default_translation, unicoded_obj.translation('XX'))