from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import client
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.command_lib.container.fleet.features import info
def get_backfill_version_from_feature(feature, membership):
    """Get the value the version field in FeatureSpec should be set to.

  Args:
    feature: the feature obtained from hub API.
    membership: The full membership name whose Spec will be backfilled.

  Returns:
    version: A string denoting the version field in MembershipConfig
  """
    spec_version, state_version = versions_for_member(feature, membership)
    if spec_version:
        return spec_version
    return state_version