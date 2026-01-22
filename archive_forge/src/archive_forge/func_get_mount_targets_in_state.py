from time import sleep
from time import time as timestamp
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_mount_targets_in_state(self, file_system_id, states=None):
    """
        Returns states of mount targets of selected EFS with selected state(s) (optional)
        """
    targets = iterate_all('MountTargets', self.connection.describe_mount_targets, FileSystemId=file_system_id)
    if states:
        if not isinstance(states, list):
            states = [states]
        targets = filter(lambda target: target['LifeCycleState'] in states, targets)
    return list(targets)