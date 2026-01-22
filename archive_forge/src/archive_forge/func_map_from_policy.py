from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
import frozendict
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.asset import client_util as asset
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api as crm
from googlecloudsdk.command_lib.artifacts import requests as artifacts
def map_from_policy(policy):
    """Converts an iam.Policy object to a map of roles to sets of users.

  Args:
    policy: An iam.Policy object

  Returns:
    A map of roles to sets of users
  """
    role_to_members = collections.defaultdict(set)
    for binding in policy.bindings:
        role_to_members[binding.role].update(binding.members)
    return role_to_members