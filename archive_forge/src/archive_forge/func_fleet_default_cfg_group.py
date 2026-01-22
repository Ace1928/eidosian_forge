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
def fleet_default_cfg_group():
    """Flag group for accepting a Fleet Default Configuration file."""
    config_group = base.ArgumentGroup('Flags for setting Fleet Default Configuration files.', mutex=True)
    config_group.AddArgument(fleet_default_cfg_flag())
    config_group.AddArgument(no_fleet_default_cfg_flag(True))
    return config_group