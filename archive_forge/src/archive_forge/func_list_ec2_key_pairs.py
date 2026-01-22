from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def list_ec2_key_pairs(connection, module):
    ids = module.params.get('ids')
    names = module.params.get('names')
    include_public_key = module.params.get('include_public_key')
    filters = module.params.get('filters')
    if filters:
        filters = ansible_dict_to_boto3_filter_list(filters)
    params = {}
    if filters:
        params['Filters'] = filters
    if ids:
        params['KeyPairIds'] = ids
    if names:
        params['KeyNames'] = names
    if include_public_key:
        params['IncludePublicKey'] = True
    try:
        result = connection.describe_key_pairs(**params)
    except is_boto3_error_code('InvalidKeyPair.NotFound'):
        result = {}
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to list EC2 key pairs')
    snaked_keys = [camel_dict_to_snake_dict(key) for key in result.get('KeyPairs', [])]
    for key in snaked_keys:
        key['tags'] = boto3_tag_list_to_ansible_dict(key.get('tags', []), 'key', 'value')
    module.exit_json(keypairs=snaked_keys)