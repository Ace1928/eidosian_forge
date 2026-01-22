from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def describe_network_acl(client, module):
    try:
        if module.params.get('nacl_id'):
            nacl = _describe_network_acls(client, Filters=[{'Name': 'network-acl-id', 'Values': [module.params.get('nacl_id')]}])
        else:
            nacl = _describe_network_acls(client, Filters=[{'Name': 'tag:Name', 'Values': [module.params.get('name')]}])
    except botocore.exceptions.ClientError as e:
        module.fail_json_aws(e)
    return nacl