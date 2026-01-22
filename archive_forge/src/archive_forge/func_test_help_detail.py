from breezy import config, i18n, tests
from breezy.tests.test_i18n import ZzzTranslations
def test_help_detail(self):
    dash_h = self.run_bzr('diff -h')[0]
    help_x = self.run_bzr('help diff')[0]
    self.assertEqual(dash_h, help_x)
    self.assertContainsRe(help_x, 'Purpose:')
    self.assertContainsRe(help_x, 'Usage:')
    self.assertContainsRe(help_x, 'Options:')
    self.assertContainsRe(help_x, 'Description:')
    self.assertContainsRe(help_x, 'Examples:')
    self.assertContainsRe(help_x, 'See also:')
    self.assertContainsRe(help_x, 'Aliases:')