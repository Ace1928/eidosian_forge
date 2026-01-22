from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
def test_recommend_upgrade_old_format(self):
    ui.ui_factory = tests.TestUIFactory()
    format = OldControlComponentFormat()
    format.check_support_status(allow_unsupported=False, recommend_upgrade=False)
    self.assertEqual('', ui.ui_factory.stderr.getvalue())
    format.check_support_status(allow_unsupported=False, recommend_upgrade=True, basedir='apath')
    self.assertEqual('An old format that is slow is deprecated and a better format is available.\nIt is recommended that you upgrade by running the command\n  brz upgrade apath\n', ui.ui_factory.stderr.getvalue())