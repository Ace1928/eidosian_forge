from collections import defaultdict
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def get_security_groups_data(self, file_systems):
    for item in file_systems:
        if item['life_cycle_state'] == self.STATE_AVAILABLE:
            for target in item['mount_targets']:
                if target['life_cycle_state'] == self.STATE_AVAILABLE:
                    try:
                        target['security_groups'] = self.get_security_groups(target['mount_target_id'])
                    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                        self.module.fail_json_aws(e, msg="Couldn't get EFS security groups")
                else:
                    target['security_groups'] = []
        else:
            item['tags'] = {}
            item['mount_targets'] = []
    return file_systems