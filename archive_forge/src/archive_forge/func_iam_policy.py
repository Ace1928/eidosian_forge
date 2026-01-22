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
def iam_policy(domain, project):
    """Generates an AR-equivalent IAM policy for a GCR registry.

  Args:
    domain: The domain of the GCR registry.
    project: The project of the GCR registry.

  Returns:
    An iam.Policy.

  Raises:
    Exception: A problem was encountered while generating the policy.
  """
    m, _ = iam_map(domain, project, skip_bucket=False, from_ar_permissions=False)
    return policy_from_map(m)