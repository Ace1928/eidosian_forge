from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import command
from googlecloudsdk.command_lib.container.fleet.policycontroller import flags
Sets the membership spec to SUSPENDED.

    Args:
      spec: The spec to be suspended.

    Returns:
      Updated spec, based on message api version.
    