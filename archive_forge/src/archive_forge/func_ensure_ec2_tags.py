import re
from ansible.module_utils.ansible_release import __version__
from ansible.module_utils.common.dict_transformations import _camel_to_snake  # pylint: disable=unused-import
from ansible.module_utils.common.dict_transformations import _snake_to_camel  # pylint: disable=unused-import
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict  # pylint: disable=unused-import
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict  # pylint: disable=unused-import
from ansible.module_utils.six import integer_types
from ansible.module_utils.six import string_types
from .arn import is_outpost_arn as is_outposts_arn  # pylint: disable=unused-import
from .botocore import HAS_BOTO3  # pylint: disable=unused-import
from .botocore import boto3_conn  # pylint: disable=unused-import
from .botocore import boto3_inventory_conn  # pylint: disable=unused-import
from .botocore import boto_exception  # pylint: disable=unused-import
from .botocore import get_aws_connection_info  # pylint: disable=unused-import
from .botocore import get_aws_region  # pylint: disable=unused-import
from .botocore import paginated_query_with_retries
from .exceptions import AnsibleAWSError  # pylint: disable=unused-import
from .modules import _aws_common_argument_spec as aws_common_argument_spec  # pylint: disable=unused-import
from .modules import aws_argument_spec as ec2_argument_spec  # pylint: disable=unused-import
from .policy import _py3cmp as py3cmp  # pylint: disable=unused-import
from .policy import compare_policies  # pylint: disable=unused-import
from .policy import sort_json_policy_dict  # pylint: disable=unused-import
from .retries import AWSRetry  # pylint: disable=unused-import
from .tagging import ansible_dict_to_boto3_tag_list  # pylint: disable=unused-import
from .tagging import boto3_tag_list_to_ansible_dict  # pylint: disable=unused-import
from .tagging import compare_aws_tags  # pylint: disable=unused-import
from .transformation import ansible_dict_to_boto3_filter_list  # pylint: disable=unused-import
from .transformation import map_complex_type  # pylint: disable=unused-import
def ensure_ec2_tags(client, module, resource_id, resource_type=None, tags=None, purge_tags=True, retry_codes=None):
    """
    Updates the tags on an EC2 resource.

    To remove all tags the tags parameter must be explicitly set to an empty dictionary.

    :param client: an EC2 boto3 client
    :param module: an AnsibleAWSModule object
    :param resource_id: the identifier for the resource
    :param resource_type: the type of the resource
    :param tags: the Tags to apply to the resource
    :param purge_tags: whether tags missing from the tag list should be removed
    :param retry_codes: additional boto3 error codes to trigger retries
    :return: changed: returns True if the tags are changed
    """
    if tags is None:
        return False
    if not retry_codes:
        retry_codes = []
    changed = False
    current_tags = describe_ec2_tags(client, module, resource_id, resource_type, retry_codes)
    tags_to_set, tags_to_unset = compare_aws_tags(current_tags, tags, purge_tags)
    if purge_tags and (not tags):
        tags_to_unset = current_tags
    changed |= remove_ec2_tags(client, module, resource_id, tags_to_unset, retry_codes)
    changed |= add_ec2_tags(client, module, resource_id, tags_to_set, retry_codes)
    return changed