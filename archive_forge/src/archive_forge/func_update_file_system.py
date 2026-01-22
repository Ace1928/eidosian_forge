from time import sleep
from time import time as timestamp
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def update_file_system(self, name, throughput_mode, provisioned_throughput_in_mibps):
    """
        Update filesystem with new throughput settings
        """
    changed = False
    state = self.get_file_system_state(name)
    if state in [self.STATE_AVAILABLE, self.STATE_CREATING]:
        fs_id = self.get_file_system_id(name)
        current_mode = self.get_throughput_mode(FileSystemId=fs_id)
        current_throughput = self.get_provisioned_throughput_in_mibps(FileSystemId=fs_id)
        params = dict()
        if throughput_mode and throughput_mode != current_mode:
            params['ThroughputMode'] = throughput_mode
        if provisioned_throughput_in_mibps and provisioned_throughput_in_mibps != current_throughput:
            params['ProvisionedThroughputInMibps'] = provisioned_throughput_in_mibps
        if len(params) > 0:
            wait_for(lambda: self.get_file_system_state(name), self.STATE_AVAILABLE, self.wait_timeout)
            try:
                self.connection.update_file_system(FileSystemId=fs_id, **params)
                changed = True
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                self.module.fail_json_aws(e, msg='Unable to update file system.')
    return changed