import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def describe_queue(client, queue_url):
    """
    Description a queue in snake format
    """
    attributes = client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=['All'], aws_retry=True)['Attributes']
    description = dict(attributes)
    description.pop('Policy', None)
    description.pop('RedrivePolicy', None)
    description = camel_dict_to_snake_dict(description)
    description['policy'] = attributes.get('Policy', None)
    description['redrive_policy'] = attributes.get('RedrivePolicy', None)
    for key, value in description.items():
        if value is None:
            continue
        if key in ['policy', 'redrive_policy']:
            policy = json.loads(value)
            description[key] = policy
            continue
        if key == 'content_based_deduplication':
            try:
                description[key] = bool(value)
            except (TypeError, ValueError):
                pass
        try:
            if value == str(int(value)):
                description[key] = int(value)
        except (TypeError, ValueError):
            pass
    return description