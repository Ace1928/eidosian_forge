from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(**backoff_params)
def replication_subnet_group_delete(module, connection):
    subnetid = module.params.get('identifier')
    delete_parameters = dict(ReplicationSubnetGroupIdentifier=subnetid)
    return connection.delete_replication_subnet_group(**delete_parameters)