import json
from traceback import format_exc
from ansible.module_utils._text import to_bytes
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def untag_secret(self, secret_name, tag_keys):
    if self.module.check_mode:
        self.module.exit_json(changed=True)
    try:
        self.client.untag_resource(SecretId=secret_name, TagKeys=tag_keys)
    except (BotoCoreError, ClientError) as e:
        self.module.fail_json_aws(e, msg='Failed to remove tag(s) from secret')