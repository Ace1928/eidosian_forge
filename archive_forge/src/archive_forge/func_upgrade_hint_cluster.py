from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import locations as api_util
from googlecloudsdk.command_lib.container.gkemulticloud import constants
def upgrade_hint_cluster(cluster_ref, cluster_info, platform):
    """Generates a message that users if their cluster version can be upgraded.

  Args:
    cluster_ref: A resource object, the cluster resource information.
    cluster_info: A GoogleCloudGkemulticloudV1AzureCluster or
      GoogleCloudGkemulticloudV1AwsCluster resource, the full list of
      information on the cluster that is to be tested.
    platform: A string, the platform the component is on {AWS,Azure}.

  Returns:
    A message in how to upgrade a cluster if its end of life.
  """
    upgrade_message = None
    valid_versions = _load_valid_versions(platform, cluster_ref.Parent())
    if _is_end_of_life(valid_versions, cluster_info.controlPlane.version):
        cluster_name = None
        if platform == constants.AWS:
            cluster_name = cluster_ref.awsClustersId
        elif platform == constants.AZURE:
            cluster_name = cluster_ref.azureClustersId
        location = cluster_ref.locationsId
        upgrade_message = _END_OF_LIFE_MESSAGE_DESCRIBE_CLUSTER
        upgrade_message += _UPGRADE_COMMAND_CLUSTER.format(USERS_PLATFORM=platform.lower(), CLUSTER_NAME=cluster_name, LOCATION=location, CLUSTER_VERSION='CLUSTER_VERSION')
        upgrade_message += _SUPPORTED_COMMAND.format(USERS_PLATFORM=platform.lower(), LOCATION=location)
    return upgrade_message