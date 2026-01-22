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
def get_sg_name(sg, boto3=None):
    return str(sg['GroupName'])