import io
import json
import yaml
from heatclient.common import format_utils
from heatclient.tests.unit.osc import utils
def test_shell_format(self):
    self.cmd = ShowShell(self.app, None)
    parsed_args = self.check_parser(self.cmd, [], [])
    expected = 'col1="abcde"\ncol2="[\'fg\', \'hi\', \'jk\']"\ncol3="{\'lmnop\': \'qrstu\'}"\n'
    self.cmd.run(parsed_args)
    self.assertEqual(expected, self.app.stdout.make_string())