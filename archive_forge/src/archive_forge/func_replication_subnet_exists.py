from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def replication_subnet_exists(subnet):
    """Returns boolean based on the existence of the endpoint
    :param endpoint: dict containing the described endpoint
    :return: bool
    """
    return bool(len(subnet['ReplicationSubnetGroups']))