import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.arn import is_outpost_arn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import describe_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def get_attachment_data(volume_dict, wanted_state=None):
    attachment_data = []
    if not volume_dict:
        return attachment_data
    resource = volume_dict.get('attachments', [])
    if wanted_state:
        resource = [data for data in resource if data['state'] == wanted_state]
    for data in resource:
        attachment_data.append({'attach_time': data.get('attach_time', None), 'device': data.get('device', None), 'instance_id': data.get('instance_id', None), 'status': data.get('state', None), 'delete_on_termination': data.get('delete_on_termination', None)})
    return attachment_data