from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def delete_static_ip(module, client, static_ip_name):
    inst = find_static_ip_info(module, client, static_ip_name)
    if inst is None:
        module.exit_json(changed=False, static_ip={})
    changed = False
    try:
        client.release_static_ip(staticIpName=static_ip_name)
        changed = True
    except botocore.exceptions.ClientError as e:
        module.fail_json_aws(e)
    module.exit_json(changed=changed, static_ip=camel_dict_to_snake_dict(inst))