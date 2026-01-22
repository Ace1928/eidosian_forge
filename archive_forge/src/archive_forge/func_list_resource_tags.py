from functools import partial
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from .retries import AWSRetry
from .tagging import boto3_tag_list_to_ansible_dict
def list_resource_tags(self, resource_arn):
    return self.client.list_tags_for_resource(Resource=resource_arn, aws_retry=True)