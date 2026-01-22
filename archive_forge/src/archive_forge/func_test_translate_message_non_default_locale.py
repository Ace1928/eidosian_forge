import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
@mock.patch('locale.getlocale')
@mock.patch('gettext.translation')
def test_translate_message_non_default_locale(self, mock_translation, mock_locale):
    message_with_params = 'A message with params: %(param)s'
    es_translation = 'A message with params in Spanish: %(param)s'
    zh_translation = 'A message with params in Chinese: %(param)s'
    fr_translation = 'A message with params in French: %(param)s'
    message_param = 'A Message param'
    es_param_translation = 'A message param in Spanish'
    zh_param_translation = 'A message param in Chinese'
    fr_param_translation = 'A message param in French'
    es_translations = {message_with_params: es_translation, message_param: es_param_translation}
    zh_translations = {message_with_params: zh_translation, message_param: zh_param_translation}
    fr_translations = {message_with_params: fr_translation, message_param: fr_param_translation}
    translator = fakes.FakeTranslations.translator({'es': es_translations, 'zh': zh_translations, 'fr': fr_translations})
    mock_translation.side_effect = translator
    mock_locale.return_value = ('es',)
    msg = _message.Message(message_with_params)
    msg_param = _message.Message(message_param)
    msg = msg % {'param': msg_param}
    es_translation = es_translation % {'param': es_param_translation}
    zh_translation = zh_translation % {'param': zh_param_translation}
    fr_translation = fr_translation % {'param': fr_param_translation}
    self.assertEqual(es_translation, msg)
    self.assertEqual(es_translation, msg.translation())
    self.assertEqual(es_translation, msg.translation('es'))
    self.assertEqual(zh_translation, msg.translation('zh'))
    self.assertEqual(fr_translation, msg.translation('fr'))