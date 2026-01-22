from uuid import uuid4
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.arn import validate_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def params_to_launch_data(module, template_params):
    if template_params.get('tags'):
        tag_list = ansible_dict_to_boto3_tag_list(template_params.get('tags'))
        template_params['tag_specifications'] = [{'resource_type': r_type, 'tags': tag_list} for r_type in ('instance', 'volume')]
        del template_params['tags']
    if module.params.get('iam_instance_profile'):
        template_params['iam_instance_profile'] = determine_iam_role(module, module.params['iam_instance_profile'])
    params = snake_dict_to_camel_dict(dict(((k, v) for k, v in template_params.items() if v is not None)), capitalize_first=True)
    return params