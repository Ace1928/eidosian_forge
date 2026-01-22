from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages as protorpc_message
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import standalone_clusters
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages_module
def standalone_node_config(self, node_config_args):
    """Constructs proto message BareMetalStandaloneNodeConfig."""
    kwargs = {'nodeIp': node_config_args.get('node-ip', ''), 'labels': self.parse_standalone_node_labels(node_config_args)}
    return self._set_config_if_exists(messages_module.BareMetalStandaloneNodeConfig, kwargs)