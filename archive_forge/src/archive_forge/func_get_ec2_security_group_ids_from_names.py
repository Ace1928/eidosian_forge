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
def get_ec2_security_group_ids_from_names(sec_group_list, ec2_connection, vpc_id=None, boto3=None):
    """Return list of security group IDs from security group names. Note that security group names are not unique
    across VPCs.  If a name exists across multiple VPCs and no VPC ID is supplied, all matching IDs will be returned. This
    will probably lead to a boto exception if you attempt to assign both IDs to a resource so ensure you wrap the call in
    a try block
    """

    def get_sg_name(sg, boto3=None):
        return str(sg['GroupName'])

    def get_sg_id(sg, boto3=None):
        return str(sg['GroupId'])
    sec_group_id_list = []
    if isinstance(sec_group_list, string_types):
        sec_group_list = [sec_group_list]
    if vpc_id:
        filters = [{'Name': 'vpc-id', 'Values': [vpc_id]}]
        all_sec_groups = ec2_connection.describe_security_groups(Filters=filters)['SecurityGroups']
    else:
        all_sec_groups = ec2_connection.describe_security_groups()['SecurityGroups']
    unmatched = set(sec_group_list).difference((str(get_sg_name(all_sg, boto3)) for all_sg in all_sec_groups))
    sec_group_name_list = list(set(sec_group_list) - set(unmatched))
    if len(unmatched) > 0:
        sec_group_id_list[:] = [sg for sg in unmatched if re.match('sg-[a-fA-F0-9]+$', sg)]
        still_unmatched = [sg for sg in unmatched if not re.match('sg-[a-fA-F0-9]+$', sg)]
        if len(still_unmatched) > 0:
            raise ValueError(f'The following group names are not valid: {', '.join(still_unmatched)}')
    sec_group_id_list += [get_sg_id(all_sg) for all_sg in all_sec_groups if get_sg_name(all_sg) in sec_group_name_list]
    return sec_group_id_list