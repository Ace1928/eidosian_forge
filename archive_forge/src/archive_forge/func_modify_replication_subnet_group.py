from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def modify_replication_subnet_group(module, connection):
    try:
        modify_params = create_module_params(module)
        return replication_subnet_group_modify(connection, **modify_params)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to Modify the DMS replication subnet group.')