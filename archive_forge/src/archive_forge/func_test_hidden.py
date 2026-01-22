from breezy import config, i18n, tests
from breezy.tests.test_i18n import ZzzTranslations
def test_hidden(self):
    help_commands = self.run_bzr('help commands')[0]
    help_hidden = self.run_bzr('help hidden-commands')[0]

    def extract_cmd_names(help_output):
        cmds = []
        for line in help_output.split('\n'):
            if line.startswith(' '):
                continue
            cmd = line.split(' ')[0]
            if line:
                cmds.append(cmd)
        return cmds
    commands = extract_cmd_names(help_commands)
    hidden = extract_cmd_names(help_hidden)
    self.assertTrue('commit' in commands)
    self.assertTrue('commit' not in hidden)
    self.assertTrue('rocks' in hidden)
    self.assertTrue('rocks' not in commands)