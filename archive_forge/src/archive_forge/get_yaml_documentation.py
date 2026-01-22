from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.container.fleet import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet import util
from googlecloudsdk.command_lib.container.fleet.memberships import errors
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
Gets P4SA account name for Anthos Support when user not specified.

    Args:
      project_id: the project ID of the resource.

    Returns:
      The P4SA account name for Anthos Support.
    