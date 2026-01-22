from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.rds import arg_spec_to_rds_params
from ansible_collections.amazon.aws.plugins.module_utils.rds import call_method
from ansible_collections.amazon.aws.plugins.module_utils.rds import ensure_tags
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.rds import wait_for_cluster_status
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
def changing_cluster_options(modify_params, current_cluster):
    changing_params = {}
    apply_immediately = modify_params.pop('ApplyImmediately')
    db_cluster_id = modify_params.pop('DBClusterIdentifier')
    enable_cloudwatch_logs_export = modify_params.pop('EnableCloudwatchLogsExports', None)
    if enable_cloudwatch_logs_export is not None:
        desired_cloudwatch_logs_configuration = {'EnableLogTypes': [], 'DisableLogTypes': []}
        provided_cloudwatch_logs = set(enable_cloudwatch_logs_export)
        current_cloudwatch_logs_export = set(current_cluster['EnabledCloudwatchLogsExports'])
        desired_cloudwatch_logs_configuration['EnableLogTypes'] = list(provided_cloudwatch_logs.difference(current_cloudwatch_logs_export))
        if module.params['purge_cloudwatch_logs_exports']:
            desired_cloudwatch_logs_configuration['DisableLogTypes'] = list(current_cloudwatch_logs_export.difference(provided_cloudwatch_logs))
        changing_params['CloudwatchLogsExportConfiguration'] = desired_cloudwatch_logs_configuration
    password = modify_params.pop('MasterUserPassword', None)
    if password:
        changing_params['MasterUserPassword'] = password
    new_cluster_id = modify_params.pop('NewDBClusterIdentifier', None)
    if new_cluster_id and new_cluster_id != current_cluster['DBClusterIdentifier']:
        changing_params['NewDBClusterIdentifier'] = new_cluster_id
    option_group = modify_params.pop('OptionGroupName', None)
    if option_group and option_group not in [g['DBClusterOptionGroupName'] for g in current_cluster['DBClusterOptionGroupMemberships']]:
        changing_params['OptionGroupName'] = option_group
    vpc_sgs = modify_params.pop('VpcSecurityGroupIds', None)
    if vpc_sgs:
        desired_vpc_sgs = []
        provided_vpc_sgs = set(vpc_sgs)
        current_vpc_sgs = set([sg['VpcSecurityGroupId'] for sg in current_cluster['VpcSecurityGroups']])
        if module.params['purge_security_groups']:
            desired_vpc_sgs = vpc_sgs
        elif provided_vpc_sgs - current_vpc_sgs:
            desired_vpc_sgs = list(provided_vpc_sgs | current_vpc_sgs)
        if desired_vpc_sgs:
            changing_params['VpcSecurityGroupIds'] = desired_vpc_sgs
    desired_db_cluster_parameter_group = modify_params.pop('DBClusterParameterGroupName', None)
    if desired_db_cluster_parameter_group:
        if desired_db_cluster_parameter_group != current_cluster['DBClusterParameterGroup']:
            changing_params['DBClusterParameterGroupName'] = desired_db_cluster_parameter_group
    for param in modify_params:
        if modify_params[param] != current_cluster[param]:
            changing_params[param] = modify_params[param]
    if changing_params:
        changing_params['DBClusterIdentifier'] = db_cluster_id
        if apply_immediately is not None:
            changing_params['ApplyImmediately'] = apply_immediately
    if module.params['state'] == 'started':
        if current_cluster['Engine'] in ['mysql', 'postgres']:
            module.fail_json('Only aurora clusters can use the state started')
        changing_params['DBClusterIdentifier'] = db_cluster_id
    if module.params['state'] == 'stopped':
        if current_cluster['Engine'] in ['mysql', 'postgres']:
            module.fail_json('Only aurora clusters can use the state stopped')
        changing_params['DBClusterIdentifier'] = db_cluster_id
    return changing_params