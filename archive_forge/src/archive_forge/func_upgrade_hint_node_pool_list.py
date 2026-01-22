from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import locations as api_util
from googlecloudsdk.command_lib.container.gkemulticloud import constants
def upgrade_hint_node_pool_list(platform, cluster_ref):
    """Generates a message that warns users if their node pool version can be upgraded.

  Args:
    platform: A string, the platform the component is on {AWS,Azure}.
    cluster_ref: A resource object, the cluster resource information.

  Returns:
    A message in how to upgrade a node pool if its end of life.
  """
    cluster_name = None
    if platform == constants.AWS:
        cluster_name = cluster_ref.awsClustersId
    elif platform == constants.AZURE:
        cluster_name = cluster_ref.azureClustersId
    upgrade_message = _UPGRADE_NODE_POOL_HINT
    upgrade_message += _UPGRADE_COMMAND_NODE_POOL.format(USERS_PLATFORM=platform.lower(), NODE_POOL_NAME='NODE_POOL_NAME', LOCATION=cluster_ref.locationsId, CLUSTER_NAME=cluster_name, NODE_POOL_VERSION='NODE_POOL_VERSION')
    upgrade_message += _SUPPORTED_COMMAND.format(USERS_PLATFORM=platform.lower(), LOCATION='LOCATION')
    return upgrade_message