from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
from ansible_collections.community.aws.plugins.module_utils.wafv2 import wafv2_list_web_acls
def list_wafv2_resources(wafv2, arn, fail_json_aws):
    try:
        response = wafv2.list_resources_for_web_acl(WebACLArn=arn)
    except (BotoCoreError, ClientError) as e:
        fail_json_aws(e, msg='Failed to list wafv2 resources.')
    return response