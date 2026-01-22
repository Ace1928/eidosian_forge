from time import sleep
from time import time as timestamp
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def converge_file_system(self, name, tags, purge_tags, targets, throughput_mode, provisioned_throughput_in_mibps):
    """
        Change attributes (mount targets and tags) of filesystem by name
        """
    result = False
    fs_id = self.get_file_system_id(name)
    if tags is not None:
        tags_need_modify, tags_to_delete = compare_aws_tags(boto3_tag_list_to_ansible_dict(self.get_tags(FileSystemId=fs_id)), tags, purge_tags)
        if tags_to_delete:
            try:
                self.connection.delete_tags(FileSystemId=fs_id, TagKeys=tags_to_delete)
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                self.module.fail_json_aws(e, msg='Unable to delete tags.')
            result = True
        if tags_need_modify:
            try:
                self.connection.create_tags(FileSystemId=fs_id, Tags=ansible_dict_to_boto3_tag_list(tags_need_modify))
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                self.module.fail_json_aws(e, msg='Unable to create tags.')
            result = True
    if targets is not None:
        incomplete_states = [self.STATE_CREATING, self.STATE_DELETING]
        wait_for(lambda: len(self.get_mount_targets_in_state(fs_id, incomplete_states)), 0)
        current_targets = _index_by_key('SubnetId', self.get_mount_targets(FileSystemId=fs_id))
        targets = _index_by_key('SubnetId', targets)
        targets_to_create, intersection, targets_to_delete = dict_diff(current_targets, targets, True)
        changed = [sid for sid in intersection if not targets_equal(['SubnetId', 'IpAddress', 'NetworkInterfaceId'], current_targets[sid], targets[sid])]
        targets_to_delete = list(targets_to_delete) + changed
        targets_to_create = list(targets_to_create) + changed
        if targets_to_delete:
            for sid in targets_to_delete:
                self.connection.delete_mount_target(MountTargetId=current_targets[sid]['MountTargetId'])
            wait_for(lambda: len(self.get_mount_targets_in_state(fs_id, incomplete_states)), 0)
            result = True
        if targets_to_create:
            for sid in targets_to_create:
                self.connection.create_mount_target(FileSystemId=fs_id, **targets[sid])
            wait_for(lambda: len(self.get_mount_targets_in_state(fs_id, incomplete_states)), 0, self.wait_timeout)
            result = True
        security_groups_to_update = [sid for sid in intersection if 'SecurityGroups' in targets[sid] and current_targets[sid]['SecurityGroups'] != targets[sid]['SecurityGroups']]
        if security_groups_to_update:
            for sid in security_groups_to_update:
                self.connection.modify_mount_target_security_groups(MountTargetId=current_targets[sid]['MountTargetId'], SecurityGroups=targets[sid].get('SecurityGroups', None))
            result = True
    return result