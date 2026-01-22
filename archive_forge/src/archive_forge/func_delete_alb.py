from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.elb_utils import get_elb_listener_rules
from ansible_collections.amazon.aws.plugins.module_utils.elbv2 import ApplicationLoadBalancer
from ansible_collections.amazon.aws.plugins.module_utils.elbv2 import ELBListener
from ansible_collections.amazon.aws.plugins.module_utils.elbv2 import ELBListenerRule
from ansible_collections.amazon.aws.plugins.module_utils.elbv2 import ELBListenerRules
from ansible_collections.amazon.aws.plugins.module_utils.elbv2 import ELBListeners
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
def delete_alb(alb_obj):
    if alb_obj.elb:
        if alb_obj.module.check_mode:
            alb_obj.module.exit_json(changed=True, msg='Would have deleted ALB if not in check mode.')
        listeners_obj = ELBListeners(alb_obj.connection, alb_obj.module, alb_obj.elb['LoadBalancerArn'])
        for listener_to_delete in [i['ListenerArn'] for i in listeners_obj.current_listeners]:
            listener_obj = ELBListener(alb_obj.connection, alb_obj.module, listener_to_delete, alb_obj.elb['LoadBalancerArn'])
            listener_obj.delete()
        alb_obj.delete()
    elif alb_obj.module.check_mode:
        alb_obj.module.exit_json(changed=False, msg='IN CHECK MODE - ALB already absent.')
    alb_obj.module.exit_json(changed=alb_obj.changed)