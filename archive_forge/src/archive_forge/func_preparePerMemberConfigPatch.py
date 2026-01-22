from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.anthos.common import file_parsers
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.identity_service import utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import retry
def preparePerMemberConfigPatch(self, args, patch, update_mask):
    membership = base.ParseMembership(args, prompt=True, autoselect=True, search=True)
    membership_spec = self.messages.MembershipFeatureSpec()
    if args.origin:
        membership_spec.origin = self.messages.Origin(type=self.messages.Origin.TypeValueValuesEnum('FLEET'))
    else:
        loaded_config = file_parsers.YamlConfigFile(file_path=args.config, item_type=file_parsers.LoginConfigObject)
        member_config = utils.parse_config(loaded_config, self.messages)
        membership_spec.identityservice = member_config
    patch.membershipSpecs = self.hubclient.ToMembershipSpecs({membership: membership_spec})
    update_mask.append('membership_specs')