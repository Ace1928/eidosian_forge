import io
import json
import os
import sys
from unittest import mock
import ddt
from osprofiler.cmd import shell
from osprofiler import exc
from osprofiler.tests import test
@mock.patch('sys.stdout', io.StringIO())
@mock.patch('osprofiler.drivers.redis_driver.Redis.get_report')
def test_trace_show_in_html(self, mock_get):
    notifications = self._create_mock_notifications()
    mock_get.return_value = notifications
    html_template = 'A long time ago in a galaxy far, far away...    some_data = $DATAIt is a period of civil war. Rebelspaceships, striking from a hiddenbase, have won their first victoryagainst the evil Galactic Empire.'
    with mock.patch('osprofiler.cmd.commands.open', mock.mock_open(read_data=html_template), create=True):
        self.run_command(self._trace_show_cmd(format_='html'))
        self.assertEqual('A long time ago in a galaxy far, far away...    some_data = %sIt is a period of civil war. Rebelspaceships, striking from a hiddenbase, have won their first victoryagainst the evil Galactic Empire.\n' % json.dumps(notifications, indent=4, separators=(',', ': ')), sys.stdout.getvalue())