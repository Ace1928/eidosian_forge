import datetime
import functools
import time
from copy import deepcopy
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def get_domain_status(client, module, domain_name):
    """
    Get the status of an existing OpenSearch cluster.
    """
    try:
        response = client.describe_domain(DomainName=domain_name)
    except is_boto3_error_code('ResourceNotFoundException'):
        return None
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg=f"Couldn't get domain {domain_name}")
    return response['DomainStatus']