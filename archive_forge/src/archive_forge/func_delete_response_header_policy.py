import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def delete_response_header_policy(self, name):
    matching_policy = self.find_response_headers_policy(name)
    if matching_policy is None:
        self.module.exit_json(msg="Didn't find a matching policy by that name, not deleting")
    else:
        policy_id = matching_policy['ResponseHeadersPolicy']['Id']
        etag = matching_policy['ETag']
        if self.check_mode:
            result = {}
        else:
            try:
                result = self.client.delete_response_headers_policy(Id=policy_id, IfMatch=etag)
            except (ClientError, BotoCoreError) as e:
                self.module.fail_json_aws(e, msg='Error deleting policy')
        self.module.exit_json(changed=True, **camel_dict_to_snake_dict(result))