import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
@mock.patch('gettext.translation')
def test_translate_multiple_languages(self, mock_translation):
    en_message = 'A message in the default locale'
    es_translation = 'A message in Spanish'
    zh_translation = 'A message in Chinese'
    message = _message.Message(en_message)
    es_translations = {en_message: es_translation}
    zh_translations = {en_message: zh_translation}
    translations_map = {'es': es_translations, 'zh': zh_translations}
    translator = fakes.FakeTranslations.translator(translations_map)
    mock_translation.side_effect = translator
    self.assertEqual(es_translation, message.translation('es'))
    self.assertEqual(zh_translation, message.translation('zh'))
    self.assertEqual(en_message, message.translation(None))
    self.assertEqual(en_message, message.translation('en'))
    self.assertEqual(en_message, message.translation('XX'))