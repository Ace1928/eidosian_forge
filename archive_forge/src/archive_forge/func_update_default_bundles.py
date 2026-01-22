from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from os import path
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet.policycontroller import protos
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.command_lib.export import util
from googlecloudsdk.core.console import console_io
def update_default_bundles(self, hub_cfg: messages.Message) -> messages.Message:
    """Sets default bundles based on args.

    This function assumes that the hub config is being initialized for the first
    time.

    Args:
      hub_cfg: A 'PolicyControllerHubConfig' proto message.

    Returns:
      A modified hub_config, adding the default bundle; or unmodified if the
      --no-default-bundles flag is specified.
    """
    if self.args.no_default_bundles:
        return hub_cfg
    policy_content_spec = self._get_policy_content(hub_cfg)
    bundles = protos.additional_properties_to_dict(policy_content_spec.bundles)
    bundles[DEFAULT_BUNDLE_NAME] = self.messages.PolicyControllerBundleInstallSpec()
    policy_content_spec.bundles = protos.set_additional_properties(self.bundle_message(), bundles)
    hub_cfg.policyContent = policy_content_spec
    return hub_cfg