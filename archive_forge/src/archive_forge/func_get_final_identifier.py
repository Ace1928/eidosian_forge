from collections import namedtuple
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
from .tagging import compare_aws_tags
from .waiters import get_waiter
def get_final_identifier(method_name, module):
    updated_identifier = None
    apply_immediately = module.params.get('apply_immediately')
    resource = get_rds_method_attribute(method_name, module).resource
    if resource == 'cluster':
        identifier = module.params['db_cluster_identifier']
        updated_identifier = module.params['new_db_cluster_identifier']
    elif resource == 'instance':
        identifier = module.params['db_instance_identifier']
        updated_identifier = module.params['new_db_instance_identifier']
    elif resource == 'instance_snapshot':
        identifier = module.params['db_snapshot_identifier']
    elif resource == 'cluster_snapshot':
        identifier = module.params['db_cluster_snapshot_identifier']
    else:
        raise NotImplementedError(f"method {method_name} hasn't been added to the list of accepted methods in module_utils/rds.py")
    if not module.check_mode and updated_identifier and apply_immediately:
        identifier = updated_identifier
    return identifier