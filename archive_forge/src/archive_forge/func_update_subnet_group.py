from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def update_subnet_group(subnet_group, name, description, subnets):
    update_params = dict()
    if description and subnet_group['description'] != description:
        update_params['Description'] = description
    if subnets:
        old_subnets = set(subnet_group['subnet_ids'])
        new_subnets = set(subnets)
        if old_subnets != new_subnets:
            update_params['SubnetIds'] = list(subnets)
    if not update_params:
        return False
    if module.check_mode:
        return True
    if 'SubnetIds' not in update_params:
        update_params['SubnetIds'] = subnet_group['subnet_ids']
    try:
        client.modify_cluster_subnet_group(aws_retry=True, ClusterSubnetGroupName=name, **update_params)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to update subnet group')
    return True