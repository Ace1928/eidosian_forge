import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
@mock.patch('gettext.translation')
def test_translate_message_with_message_parameters(self, mock_translation):
    message_with_params = 'A message with params: %s %s'
    es_translation = 'A message with params in Spanish: %s %s'
    message_param = 'A message param'
    es_param_translation = 'A message param in Spanish'
    another_message_param = 'Another message param'
    another_es_param_translation = 'Another message param in Spanish'
    translations = {message_with_params: es_translation, message_param: es_param_translation, another_message_param: another_es_param_translation}
    translator = fakes.FakeTranslations.translator({'es': translations})
    mock_translation.side_effect = translator
    msg = _message.Message(message_with_params)
    param_1 = _message.Message(message_param)
    param_2 = _message.Message(another_message_param)
    msg = msg % (param_1, param_2)
    default_translation = message_with_params % (message_param, another_message_param)
    expected_translation = es_translation % (es_param_translation, another_es_param_translation)
    self.assertEqual(expected_translation, msg.translation('es'))
    self.assertEqual(default_translation, msg.translation('XX'))