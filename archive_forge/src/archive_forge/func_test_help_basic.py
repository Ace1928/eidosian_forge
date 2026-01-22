from breezy import config, i18n, tests
from breezy.tests.test_i18n import ZzzTranslations
def test_help_basic(self):
    for cmd in ['--help', 'help', '-h', '-?']:
        output = self.run_bzr(cmd)[0]
        line1 = output.split('\n')[0]
        if not line1.startswith('Breezy'):
            self.fail('bad output from brz {}:\n{!r}'.format(cmd, output))