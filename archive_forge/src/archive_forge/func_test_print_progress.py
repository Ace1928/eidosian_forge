import argparse
import base64
import builtins
import collections
import datetime
import io
import os
from unittest import mock
import fixtures
from oslo_utils import timeutils
import testtools
import novaclient
from novaclient import api_versions
from novaclient import base
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
import novaclient.v2.shell
@mock.patch('novaclient.v2.shell.sys.stdout')
@mock.patch('novaclient.v2.shell.time')
def test_print_progress(self, mock_time, mock_stdout):
    updated_objects = (base.Resource(None, info={'status': 'INPROGRESS', 'progress': 0}), base.Resource(None, info={'status': 'INPROGRESS', 'progress': 50}), base.Resource(None, info={'status': 'OK', 'progress': 100}))
    poll_fn = mock.MagicMock(side_effect=updated_objects)
    action = 'some'
    novaclient.v2.shell._poll_for_status(poll_fn=poll_fn, obj_id='uuuuuuuuuuuiiiiiiiii', final_ok_states=['ok'], poll_period='3', action=action, show_progress=True, silent=False)
    stdout_arg_list = [mock.call('\n'), mock.call('\rServer %s... 0%% complete' % action), mock.call('\rServer %s... 50%% complete' % action), mock.call('\rServer %s... 100%% complete' % action), mock.call('\nFinished'), mock.call('\n')]
    self.assertEqual(stdout_arg_list, mock_stdout.write.call_args_list)