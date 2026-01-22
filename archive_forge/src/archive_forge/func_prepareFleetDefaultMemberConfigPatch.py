from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.anthos.common import file_parsers
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.identity_service import utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import retry
def prepareFleetDefaultMemberConfigPatch(self, args, patch, update_mask):
    loaded_config = file_parsers.YamlConfigFile(file_path=args.fleet_default_member_config, item_type=file_parsers.LoginConfigObject)
    member_config = utils.parse_config(loaded_config, self.messages)
    patch.fleetDefaultMemberConfig = self.messages.CommonFleetDefaultMemberConfigSpec(identityservice=member_config)
    update_mask.append('fleet_default_member_config')