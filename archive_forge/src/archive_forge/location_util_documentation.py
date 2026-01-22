from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.fault_injection import constants
from googlecloudsdk.core.console import console_io
Prompt for Location from list of available locations.

  This method is referenced by the declaritive iam commands as a fallthrough
  for getting the location.

  Args:
    available_locations: list of the available locations to choose from

  Returns:
    The location specified by the user, str
  