from time import sleep
from time import time as timestamp
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_file_systems(self, **kwargs):
    """
        Returns generator of file systems including all attributes of FS
        """
    items = iterate_all('FileSystems', self.connection.describe_file_systems, **kwargs)
    for item in items:
        item['Name'] = item['CreationToken']
        item['CreationTime'] = str(item['CreationTime'])
        '\n            In the time when MountPoint was introduced there was a need to add a suffix of network path before one could use it\n            AWS updated it and now there is no need to add a suffix. MountPoint is left for back-compatibility purpose\n            And new FilesystemAddress variable is introduced for direct use with other modules (e.g. mount)\n            AWS documentation is available here:\n            https://docs.aws.amazon.com/efs/latest/ug/gs-step-three-connect-to-ec2-instance.html\n            '
        item['MountPoint'] = f'.{item['FileSystemId']}.efs.{self.region}.amazonaws.com:/'
        item['FilesystemAddress'] = f'{item['FileSystemId']}.efs.{self.region}.amazonaws.com:/'
        if 'Timestamp' in item['SizeInBytes']:
            item['SizeInBytes']['Timestamp'] = str(item['SizeInBytes']['Timestamp'])
        if item['LifeCycleState'] == self.STATE_AVAILABLE:
            item['Tags'] = self.get_tags(FileSystemId=item['FileSystemId'])
            item['MountTargets'] = list(self.get_mount_targets(FileSystemId=item['FileSystemId']))
        else:
            item['Tags'] = {}
            item['MountTargets'] = []
        yield item