from breezy import config, i18n, tests
from breezy.tests.test_i18n import ZzzTranslations
def test_help_checkouts(self):
    """Smoke test for 'brz help checkouts'"""
    out, err = self.run_bzr('help checkouts')
    self.assertContainsRe(out, 'checkout')
    self.assertContainsRe(out, 'lightweight')