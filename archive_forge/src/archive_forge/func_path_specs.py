from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet import util as fleet_util
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.core import exceptions as gcloud_exceptions
import six
def path_specs(self, args: parser_extensions.Namespace, ignore_missing: bool=False, ignore_metadata: bool=True) -> SpecMapping:
    """Retrieves memberships specified by the command that exist in the Feature.

    Args:
      args: The argparse object passed to the command.
      ignore_missing: Use this to return a mapping that includes an 'empty' spec
        for each specified path if it doesn't already exist.
      ignore_metadata: If true, remove the Hub-managed metadata (i.e. origin).
        If the spec is being retrieved for reporting (i.e. describe) then set to
        false to get the full current value. If it is being used to update the
        spec leave as True so that the return spec can be used in a patch.

    Returns:
      A dict mapping a path to the membership spec.

    Raises:
      exceptions.DisabledMembershipError: If the membership is invalid or not
      enabled.
    """
    memberships_paths = self._membership_paths(args)
    specs = {fleet_util.MembershipPartialName(path): (path, spec) for path, spec in self.current_specs().items() if fleet_util.MembershipPartialName(path) in memberships_paths}
    if ignore_metadata:
        specs = {partial_path: (path, self._rebuild_spec(spec)) for partial_path, (path, spec) in specs.items()}
    if ignore_missing:
        missing = [(s, f) for s, f in memberships_paths.items() if s not in specs]
        for short, full in missing:
            specs[short] = (full, self.messages.MembershipFeatureSpec())
    else:
        msg = 'Policy Controller is not enabled for membership {}'
        missing_memberships = [exceptions.InvalidPocoMembershipError(msg.format(path)) for path in memberships_paths if path not in specs]
        if missing_memberships:
            raise exceptions.InvalidPocoMembershipError(missing_memberships)
    return {path: spec for path, spec in specs.values()}