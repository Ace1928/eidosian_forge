from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base as feature_base
from googlecloudsdk.core import log
def update_policy_controller_state(self, md):
    """Update policy controller state for the membership that has ACM installed.

    Args:
      md: MembershipFeatureState
    """
    if md.state.code.name != 'OK':
        self.policy_controller_state = 'ERROR: {}'.format(md.state.description)
        return
    fs = md.configmanagement
    if not (fs.policyControllerState and fs.policyControllerState.deploymentState):
        self.policy_controller_state = NA
        return
    pc_deployment_state = fs.policyControllerState.deploymentState
    expected_deploys = {'GatekeeperControllerManager': pc_deployment_state.gatekeeperControllerManagerState}
    if fs.membershipSpec and fs.membershipSpec.version and (fs.membershipSpec.version > '1.4.1'):
        expected_deploys['GatekeeperAudit'] = pc_deployment_state.gatekeeperAudit
    for deployment_name, deployment_state in expected_deploys.items():
        if not deployment_state:
            continue
        elif deployment_state.name != 'INSTALLED':
            self.policy_controller_state = '{} {}'.format(deployment_name, deployment_state)
            return
        self.policy_controller_state = deployment_state.name