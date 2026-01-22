import time
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.arn import parse_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.base import BaseResourceManager
from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
from ansible_collections.community.aws.plugins.module_utils.base import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
def set_domain_list(self, options):
    if not options:
        return False
    changed = False
    domain_names = options.get('domain_names')
    home_net = options.get('source_ips', None)
    action = options.get('action')
    filter_http = options.get('filter_http', False)
    filter_https = options.get('filter_https', False)
    if home_net:
        changed |= self.set_ip_variables(dict(HOME_NET=home_net), purge=True)
    else:
        self.set_ip_variables(dict(), purge=True)
    target_types = []
    if filter_http:
        target_types.append('HTTP_HOST')
    if filter_https:
        target_types.append('TLS_SNI')
    if action == 'allow':
        action = 'ALLOWLIST'
    else:
        action = 'DENYLIST'
    rule = dict(Targets=domain_names, TargetTypes=target_types, GeneratedRulesType=action)
    changed |= self._set_rule_source('RulesSourceList', rule)
    return changed