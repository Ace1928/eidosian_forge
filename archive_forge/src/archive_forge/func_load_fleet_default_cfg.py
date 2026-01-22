from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from os import path
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet.policycontroller import protos
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.command_lib.export import util
from googlecloudsdk.core.console import console_io
def load_fleet_default_cfg(self) -> messages.Message:
    if self.args.fleet_default_member_config:
        config_path = path.expanduser(self.args.fleet_default_member_config)
        data = console_io.ReadFromFileOrStdin(config_path, binary=False)
        return util.Import(self.messages.PolicyControllerMembershipSpec, data)