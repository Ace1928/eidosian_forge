import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def convert_grant_params(grant, key):
    grant_params = dict(KeyId=key['key_arn'], GranteePrincipal=grant['grantee_principal'])
    if grant.get('operations'):
        grant_params['Operations'] = grant['operations']
    if grant.get('retiring_principal'):
        grant_params['RetiringPrincipal'] = grant['retiring_principal']
    if grant.get('name'):
        grant_params['Name'] = grant['name']
    if grant.get('constraints'):
        grant_params['Constraints'] = dict()
        if grant['constraints'].get('encryption_context_subset'):
            grant_params['Constraints']['EncryptionContextSubset'] = grant['constraints']['encryption_context_subset']
        if grant['constraints'].get('encryption_context_equals'):
            grant_params['Constraints']['EncryptionContextEquals'] = grant['constraints']['encryption_context_equals']
    return grant_params