from oslotest import base as test_base
import oslo_i18n
from oslo_i18n import _gettextutils
from oslo_i18n._i18n import _
from oslo_i18n import _lazy
from oslo_i18n import _message
from oslo_i18n import _translate
from oslo_i18n import fixture
def test_extra_lang(self):
    languages = _gettextutils.get_available_languages('oslo')
    languages.append(_FAKE_LANG)
    self.useFixture(fixture.PrefixLazyTranslation(languages=languages))
    raw_id1 = 'fake msg1'
    expected_msg_en_US = 'oslo_i18n/' + fixture.PrefixLazyTranslation._DEFAULT_LANG + ': ' + raw_id1
    expected_msg_en_ZZ = 'oslo_i18n/' + _FAKE_LANG + ': ' + raw_id1
    msg1 = _(raw_id1)
    self.assertEqual(languages, _gettextutils.get_available_languages('oslo_i18n'))
    self.assertEqual(languages, oslo_i18n.get_available_languages('oslo_i18n'))
    self.assertEqual(expected_msg_en_US, _translate.translate(msg1))
    self.assertEqual(expected_msg_en_ZZ, _translate.translate(msg1, desired_locale=_FAKE_LANG))