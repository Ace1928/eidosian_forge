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
def update_monitoring(self, hub_cfg: messages.Message) -> messages.Message:
    """Sets or removes monitoring backends based on args."""
    if self.args.no_monitoring:
        config = self.messages.PolicyControllerMonitoringConfig(backends=[])
        hub_cfg.monitoring = config
    if self.args.monitoring:
        backends = [self._get_monitoring_enum(backend) for backend in self.args.monitoring.split(',')]
        config = self.messages.PolicyControllerMonitoringConfig(backends=backends)
        hub_cfg.monitoring = config
    return hub_cfg