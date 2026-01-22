from breezy import config, i18n, tests
from breezy.tests.test_i18n import ZzzTranslations
def test_help_working_trees(self):
    """Smoke test for 'brz help working-trees'"""
    out, err = self.run_bzr('help working-trees')
    from breezy.help_topics import _working_trees, help_as_plain_text
    expected = help_as_plain_text(_working_trees)
    self.assertEqual(expected, out)