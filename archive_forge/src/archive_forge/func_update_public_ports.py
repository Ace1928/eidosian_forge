import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def update_public_ports(module, client, instance_name):
    try:
        client.put_instance_public_ports(portInfos=snake_dict_to_camel_dict(module.params.get('public_ports')), instanceName=instance_name)
    except botocore.exceptions.ClientError as e:
        module.fail_json_aws(e)