from breezy import config, i18n, tests
from breezy.tests.test_i18n import ZzzTranslations
def test_help_with_aliases(self):
    original = self.run_bzr('help cat')[0]
    conf = config.GlobalConfig.from_string('[ALIASES]\nc=cat\ncat=cat\n', save=True)
    expected = original + "'brz cat' is an alias for 'brz cat'.\n"
    self.assertEqual(expected, self.run_bzr('help cat')[0])
    self.assertEqual("'brz c' is an alias for 'brz cat'.\n", self.run_bzr('help c')[0])