import time
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import map_complex_type
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def is_matching_service(self, expected, existing):
    if expected['task_definition'] != existing['taskDefinition'].split('/')[-1]:
        if existing.get('deploymentController', {}).get('type', None) != 'CODE_DEPLOY':
            return False
    if expected.get('health_check_grace_period_seconds'):
        if expected.get('health_check_grace_period_seconds') != existing.get('healthCheckGracePeriodSeconds'):
            return False
    if (expected['load_balancers'] or []) != existing['loadBalancers']:
        return False
    if (expected['propagate_tags'] or 'NONE') != existing['propagateTags']:
        return False
    if boto3_tag_list_to_ansible_dict(existing.get('tags', [])) != (expected['tags'] or {}):
        return False
    if (expected['enable_execute_command'] or False) != existing.get('enableExecuteCommand', False):
        return False
    if expected['scheduling_strategy'] != 'DAEMON':
        if (expected['desired_count'] or 0) != existing['desiredCount']:
            return False
    return True