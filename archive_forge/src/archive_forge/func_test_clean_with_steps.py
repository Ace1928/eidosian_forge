import copy
import io
import json
import sys
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient.common import utils as commonutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_node
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
from ironicclient.v1 import utils as v1_utils
def test_clean_with_steps(self):
    steps_dict = {'clean_steps': [{'interface': 'raid', 'step': 'create_configuration', 'args': {'create_nonroot_volumes': False}}, {'interface': 'deploy', 'step': 'erase_devices'}]}
    steps_json = json.dumps(steps_dict)
    arglist = ['--clean-steps', steps_json, 'node_uuid']
    verifylist = [('clean_steps', steps_json), ('provision_state', 'clean'), ('nodes', ['node_uuid'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.node.set_provision_state.assert_called_once_with('node_uuid', 'clean', cleansteps=steps_dict, configdrive=None, deploysteps=None, rescue_password=None)