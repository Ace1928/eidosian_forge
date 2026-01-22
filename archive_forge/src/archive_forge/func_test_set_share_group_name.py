import argparse
from unittest import mock
import uuid
from osc_lib import exceptions
from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as oscutils
from manilaclient.osc import utils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_groups as osc_share_groups
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_set_share_group_name(self):
    new_name = uuid.uuid4().hex
    arglist = ['--name', new_name, self.share_group.id]
    verifylist = [('name', new_name), ('share_group', self.share_group.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.groups_mock.update.assert_called_with(self.share_group.id, name=parsed_args.name)