from collections import namedtuple
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
from .tagging import compare_aws_tags
from .waiters import get_waiter
def get_rds_method_attribute(method_name, module):
    """
    Returns rds attributes of the specified method.

        Parameters:
            method_name (str): RDS method to call
            module: AnsibleAWSModule

        Returns:
            Boto3ClientMethod (dict):
                name (str): Name of method
                waiter (str): Name of waiter associated with given method
                operation_description (str): Description of method
                resource (str): Type of resource this method applies to
                                One of ['instance', 'cluster', 'instance_snapshot', 'cluster_snapshot']
                retry_codes (list): List of extra error codes to retry on

        Raises:
            NotImplementedError if wait is True but no waiter can be found for specified method
    """
    waiter = ''
    readable_op = method_name.replace('_', ' ').replace('db', 'DB')
    resource = ''
    retry_codes = []
    if method_name in cluster_method_names and 'new_db_cluster_identifier' in module.params:
        resource = 'cluster'
        if method_name == 'delete_db_cluster':
            waiter = 'cluster_deleted'
        else:
            waiter = 'cluster_available'
        if method_name == 'restore_db_cluster_from_snapshot':
            retry_codes = ['InvalidDBClusterSnapshotState']
        else:
            retry_codes = ['InvalidDBClusterState']
    elif method_name in instance_method_names and 'new_db_instance_identifier' in module.params:
        resource = 'instance'
        if method_name == 'delete_db_instance':
            waiter = 'db_instance_deleted'
        elif method_name == 'stop_db_instance':
            waiter = 'db_instance_stopped'
        elif method_name == 'add_role_to_db_instance':
            waiter = 'role_associated'
        elif method_name == 'remove_role_from_db_instance':
            waiter = 'role_disassociated'
        elif method_name == 'promote_read_replica':
            waiter = 'read_replica_promoted'
        elif method_name == 'db_cluster_promoting':
            waiter = 'db_cluster_promoting'
        else:
            waiter = 'db_instance_available'
        if method_name == 'restore_db_instance_from_db_snapshot':
            retry_codes = ['InvalidDBSnapshotState']
        else:
            retry_codes = ['InvalidDBInstanceState', 'InvalidDBSecurityGroupState']
    elif method_name in cluster_snapshot_method_names and 'db_cluster_snapshot_identifier' in module.params:
        resource = 'cluster_snapshot'
        if method_name == 'delete_db_cluster_snapshot':
            waiter = 'db_cluster_snapshot_deleted'
            retry_codes = ['InvalidDBClusterSnapshotState']
        elif method_name == 'create_db_cluster_snapshot':
            waiter = 'db_cluster_snapshot_available'
            retry_codes = ['InvalidDBClusterState']
        else:
            waiter = 'db_cluster_snapshot_available'
            retry_codes = ['InvalidDBClusterSnapshotState']
    elif method_name in instance_snapshot_method_names and 'db_snapshot_identifier' in module.params:
        resource = 'instance_snapshot'
        if method_name == 'delete_db_snapshot':
            waiter = 'db_snapshot_deleted'
            retry_codes = ['InvalidDBSnapshotState']
        elif method_name == 'create_db_snapshot':
            waiter = 'db_snapshot_available'
            retry_codes = ['InvalidDBInstanceState']
        else:
            waiter = 'db_snapshot_available'
            retry_codes = ['InvalidDBSnapshotState']
    elif module.params.get('wait'):
        raise NotImplementedError(f"method {method_name} hasn't been added to the list of accepted methods to use a waiter in module_utils/rds.py")
    return Boto3ClientMethod(name=method_name, waiter=waiter, operation_description=readable_op, resource=resource, retry_codes=retry_codes)