from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import client
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.command_lib.container.fleet.features import info
def versions_for_member(feature, membership):
    """Parses the version fields from an ACM Feature for a given membership.

  Args:
    feature: A v1alpha, v1beta, or v1 ACM Feature.
    membership: The full membership name whose version to return.

  Returns:
    A tuple of the form (spec.version, state.spec.version), with unset versions
    defaulting to the empty string.
  """
    spec_version = None
    specs = client.HubClient.ToPyDict(feature.membershipSpecs)
    for full_membership, spec in specs.items():
        if util.MembershipPartialName(full_membership) == util.MembershipPartialName(membership):
            if spec is not None and spec.configmanagement is not None:
                spec_version = spec.configmanagement.version
            break
    state_version = None
    states = client.HubClient.ToPyDict(feature.membershipStates)
    for full_membership, state in states.items():
        if util.MembershipPartialName(full_membership) == util.MembershipPartialName(membership):
            if state is not None and state.configmanagement is not None:
                if state.configmanagement.membershipSpec is not None:
                    state_version = state.configmanagement.membershipSpec.version
            break
    return (spec_version or '', state_version or '')