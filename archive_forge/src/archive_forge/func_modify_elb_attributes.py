import traceback
from copy import deepcopy
from .ec2 import get_ec2_security_group_ids_from_names
from .elb_utils import convert_tg_name_to_arn
from .elb_utils import get_elb
from .elb_utils import get_elb_listener
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
from .waiters import get_waiter
def modify_elb_attributes(self):
    """
        Update Network ELB attributes if required

        :return:
        """
    update_attributes = []
    if self.cross_zone_load_balancing is not None and str(self.cross_zone_load_balancing).lower() != self.elb_attributes['load_balancing_cross_zone_enabled']:
        update_attributes.append({'Key': 'load_balancing.cross_zone.enabled', 'Value': str(self.cross_zone_load_balancing).lower()})
    if self.deletion_protection is not None and str(self.deletion_protection).lower() != self.elb_attributes['deletion_protection_enabled']:
        update_attributes.append({'Key': 'deletion_protection.enabled', 'Value': str(self.deletion_protection).lower()})
    if update_attributes:
        try:
            AWSRetry.jittered_backoff()(self.connection.modify_load_balancer_attributes)(LoadBalancerArn=self.elb['LoadBalancerArn'], Attributes=update_attributes)
            self.changed = True
        except (BotoCoreError, ClientError) as e:
            if self.new_load_balancer:
                AWSRetry.jittered_backoff()(self.connection.delete_load_balancer)(LoadBalancerArn=self.elb['LoadBalancerArn'])
            self.module.fail_json_aws(e)