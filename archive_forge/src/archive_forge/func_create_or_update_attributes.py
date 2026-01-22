import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_or_update_attributes(connection, module, target_group, new_target_group):
    changed = False
    target_type = module.params.get('target_type')
    deregistration_delay_timeout = module.params.get('deregistration_delay_timeout')
    deregistration_connection_termination = module.params.get('deregistration_connection_termination')
    stickiness_enabled = module.params.get('stickiness_enabled')
    stickiness_lb_cookie_duration = module.params.get('stickiness_lb_cookie_duration')
    stickiness_type = module.params.get('stickiness_type')
    stickiness_app_cookie_duration = module.params.get('stickiness_app_cookie_duration')
    stickiness_app_cookie_name = module.params.get('stickiness_app_cookie_name')
    preserve_client_ip_enabled = module.params.get('preserve_client_ip_enabled')
    proxy_protocol_v2_enabled = module.params.get('proxy_protocol_v2_enabled')
    load_balancing_algorithm_type = module.params.get('load_balancing_algorithm_type')
    update_attributes = []
    current_tg_attributes = get_tg_attributes(connection, module, target_group['TargetGroupArn'])
    if deregistration_delay_timeout is not None:
        if str(deregistration_delay_timeout) != current_tg_attributes['deregistration_delay_timeout_seconds']:
            update_attributes.append({'Key': 'deregistration_delay.timeout_seconds', 'Value': str(deregistration_delay_timeout)})
    if deregistration_connection_termination is not None:
        if deregistration_connection_termination and current_tg_attributes.get('deregistration_delay_connection_termination_enabled') != 'true':
            update_attributes.append({'Key': 'deregistration_delay.connection_termination.enabled', 'Value': 'true'})
    if stickiness_enabled is not None:
        if stickiness_enabled and current_tg_attributes['stickiness_enabled'] != 'true':
            update_attributes.append({'Key': 'stickiness.enabled', 'Value': 'true'})
    if stickiness_lb_cookie_duration is not None:
        if str(stickiness_lb_cookie_duration) != current_tg_attributes['stickiness_lb_cookie_duration_seconds']:
            update_attributes.append({'Key': 'stickiness.lb_cookie.duration_seconds', 'Value': str(stickiness_lb_cookie_duration)})
    if stickiness_type is not None:
        if stickiness_type != current_tg_attributes.get('stickiness_type'):
            update_attributes.append({'Key': 'stickiness.type', 'Value': stickiness_type})
    if stickiness_app_cookie_name is not None:
        if stickiness_app_cookie_name != current_tg_attributes.get('stickiness_app_cookie_name'):
            update_attributes.append({'Key': 'stickiness.app_cookie.cookie_name', 'Value': str(stickiness_app_cookie_name)})
    if stickiness_app_cookie_duration is not None:
        if str(stickiness_app_cookie_duration) != current_tg_attributes['stickiness_app_cookie_duration_seconds']:
            update_attributes.append({'Key': 'stickiness.app_cookie.duration_seconds', 'Value': str(stickiness_app_cookie_duration)})
    if preserve_client_ip_enabled is not None:
        if target_type not in ('udp', 'tcp_udp'):
            if str(preserve_client_ip_enabled).lower() != current_tg_attributes.get('preserve_client_ip_enabled'):
                update_attributes.append({'Key': 'preserve_client_ip.enabled', 'Value': str(preserve_client_ip_enabled).lower()})
    if proxy_protocol_v2_enabled is not None:
        if str(proxy_protocol_v2_enabled).lower() != current_tg_attributes.get('proxy_protocol_v2_enabled'):
            update_attributes.append({'Key': 'proxy_protocol_v2.enabled', 'Value': str(proxy_protocol_v2_enabled).lower()})
    if load_balancing_algorithm_type is not None:
        if str(load_balancing_algorithm_type) != current_tg_attributes['load_balancing_algorithm_type']:
            update_attributes.append({'Key': 'load_balancing.algorithm.type', 'Value': str(load_balancing_algorithm_type)})
    if update_attributes:
        try:
            connection.modify_target_group_attributes(TargetGroupArn=target_group['TargetGroupArn'], Attributes=update_attributes, aws_retry=True)
            changed = True
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            if new_target_group:
                connection.delete_target_group(TargetGroupArn=target_group['TargetGroupArn'], aws_retry=True)
            module.fail_json_aws(e, msg="Couldn't delete target group")
    return changed