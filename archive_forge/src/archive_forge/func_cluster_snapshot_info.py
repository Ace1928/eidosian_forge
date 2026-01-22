from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def cluster_snapshot_info(module, conn):
    snapshot_name = module.params.get('db_cluster_snapshot_identifier')
    snapshot_type = module.params.get('snapshot_type')
    instance_name = module.params.get('db_cluster_identifier')
    params = dict()
    if snapshot_name:
        params['DBClusterSnapshotIdentifier'] = snapshot_name
    if instance_name:
        params['DBClusterIdentifier'] = instance_name
    if snapshot_type:
        params['SnapshotType'] = snapshot_type
        if snapshot_type == 'public':
            params['IncludePublic'] = True
        elif snapshot_type == 'shared':
            params['IncludeShared'] = True
    return common_snapshot_info(module, conn, 'describe_db_cluster_snapshots', 'DBClusterSnapshot', params)