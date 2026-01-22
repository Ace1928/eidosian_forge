from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def get_bucket_details(connection, name, requested_facts, transform_location):
    """
    Execute all enabled S3API get calls for selected bucket
    """
    all_facts = {}
    for key in requested_facts:
        if requested_facts[key]:
            if key == 'bucket_location':
                all_facts[key] = {}
                try:
                    all_facts[key] = get_bucket_location(name, connection, transform_location)
                except botocore.exceptions.ClientError:
                    pass
            elif key == 'bucket_tagging':
                all_facts[key] = {}
                try:
                    all_facts[key] = get_bucket_tagging(name, connection)
                except botocore.exceptions.ClientError:
                    pass
            else:
                all_facts[key] = {}
                try:
                    all_facts[key] = get_bucket_property(name, connection, key)
                except botocore.exceptions.ClientError:
                    pass
    return all_facts