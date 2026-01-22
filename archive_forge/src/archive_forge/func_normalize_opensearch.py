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
def normalize_opensearch(client, module, domain):
    """
    Merge the input domain object with tags associated with the domain,
    convert the attributes from camel case to snake case, and return the object.
    """
    try:
        domain['Tags'] = boto3_tag_list_to_ansible_dict(client.list_tags(ARN=domain['ARN'], aws_retry=True)['TagList'])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, f"Couldn't get tags for domain {domain['domain_name']}")
    except KeyError:
        module.fail_json(msg=str(domain))
    return camel_dict_to_snake_dict(domain, ignore_list=['Tags'])