from breezy import config, i18n, tests
from breezy.tests.test_i18n import ZzzTranslations
def test_help_command_ascii(self):
    out, err = self.run_bzr_raw(['help', 'push'], encoding='ascii')
    self.assertContainsRe(out, b'zz\\?{{:See also:')