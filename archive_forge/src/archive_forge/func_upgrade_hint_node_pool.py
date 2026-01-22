from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import locations as api_util
from googlecloudsdk.command_lib.container.gkemulticloud import constants
def upgrade_hint_node_pool(node_pool_ref, node_pool_info, platform):
    """Generates a message that users if their node pool version can be upgraded.

  Args:
    node_pool_ref: A resource object, the node pool resource information.
    node_pool_info: A GoogleCloudGkemulticloudV1AzureNodePool or
      GoogleCloudGkemulticloudV1AwsNodePool resource, the full list of
      information on the node pool that is to be tested.
    platform: A string, the platform the component is on {AWS,Azure}.

  Returns:
    A message in how to upgrade a node pool if its end of life.
  """
    upgrade_message = None
    valid_versions = _load_valid_versions(platform, node_pool_ref.Parent().Parent())
    if not _is_end_of_life(valid_versions, node_pool_info.version):
        return upgrade_message
    cluster_name = None
    node_pool_name = None
    if platform == constants.AWS:
        cluster_name = node_pool_ref.awsClustersId
        node_pool_name = node_pool_ref.awsNodePoolsId
    elif platform == constants.AZURE:
        cluster_name = node_pool_ref.azureClustersId
        node_pool_name = node_pool_ref.azureNodePoolsId
    location = node_pool_ref.locationsId
    upgrade_message = _END_OF_LIFE_MESSAGE_DESCRIBE_NODE_POOL
    upgrade_message += _UPGRADE_COMMAND_NODE_POOL.format(USERS_PLATFORM=platform.lower(), NODE_POOL_NAME=node_pool_name, LOCATION=location, CLUSTER_NAME=cluster_name, NODE_POOL_VERSION='NODE_POOL_VERSION')
    upgrade_message += _SUPPORTED_COMMAND.format(USERS_PLATFORM=platform.lower(), LOCATION=location)
    return upgrade_message