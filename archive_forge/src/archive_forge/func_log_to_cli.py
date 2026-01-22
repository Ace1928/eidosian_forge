import copy
import itertools
import json
import logging
import os
import time
from collections import Counter
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional, Set, Tuple
import boto3
import botocore
from packaging.version import Version
from ray.autoscaler._private.aws.cloudwatch.cloudwatch_helper import (
from ray.autoscaler._private.aws.utils import (
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.providers import _PROVIDER_PRETTY_NAMES
from ray.autoscaler._private.util import check_legacy_fields
from ray.autoscaler.tags import NODE_TYPE_LEGACY_HEAD, NODE_TYPE_LEGACY_WORKER
def log_to_cli(config: Dict[str, Any]) -> None:
    provider_name = _PROVIDER_PRETTY_NAMES.get('aws', None)
    cli_logger.doassert(provider_name is not None, 'Could not find a pretty name for the AWS provider.')
    head_node_type = config['head_node_type']
    head_node_config = config['available_node_types'][head_node_type]['node_config']
    with cli_logger.group('{} config', provider_name):

        def print_info(resource_string: str, key: str, src_key: str, allowed_tags: Optional[List[str]]=None, list_value: bool=False) -> None:
            if allowed_tags is None:
                allowed_tags = ['default']
            node_tags = {}
            unique_settings = set()
            for node_type_key, node_type in config['available_node_types'].items():
                node_tags[node_type_key] = {}
                tag = _log_info[src_key][node_type_key]
                if tag in allowed_tags:
                    node_tags[node_type_key][tag] = True
                setting = node_type['node_config'].get(key)
                if list_value:
                    unique_settings.add(tuple(setting))
                else:
                    unique_settings.add(setting)
            head_value_str = head_node_config[key]
            if list_value:
                head_value_str = cli_logger.render_list(head_value_str)
            if len(unique_settings) == 1:
                cli_logger.labeled_value(resource_string + ' (all available node types)', '{}', head_value_str, _tags=node_tags[config['head_node_type']])
            else:
                cli_logger.labeled_value(resource_string + f' ({head_node_type})', '{}', head_value_str, _tags=node_tags[head_node_type])
                for node_type_key, node_type in config['available_node_types'].items():
                    if node_type_key == head_node_type:
                        continue
                    workers_value_str = node_type['node_config'][key]
                    if list_value:
                        workers_value_str = cli_logger.render_list(workers_value_str)
                    cli_logger.labeled_value(resource_string + f' ({node_type_key})', '{}', workers_value_str, _tags=node_tags[node_type_key])
        tags = {'default': _log_info['head_instance_profile_src'] == 'default'}
        assert 'IamInstanceProfile' in head_node_config or 'IamInstanceProfile' in config['head_node']
        if 'IamInstanceProfile' in head_node_config:
            IamProfile = head_node_config['IamInstanceProfile']
        elif 'IamInstanceProfile' in config['head_node']:
            IamProfile = config['head_node']['IamInstanceProfile']
        profile_arn = IamProfile.get('Arn')
        profile_name = _arn_to_name(profile_arn) if profile_arn else IamProfile['Name']
        cli_logger.labeled_value('IAM Profile', '{}', profile_name, _tags=tags)
        if all(('KeyName' in node_type['node_config'] for node_type in config['available_node_types'].values())):
            print_info('EC2 Key pair', 'KeyName', 'keypair_src')
        print_info('VPC Subnets', 'SubnetIds', 'subnet_src', list_value=True)
        print_info('EC2 Security groups', 'SecurityGroupIds', 'security_group_src', list_value=True)
        print_info('EC2 AMI', 'ImageId', 'ami_src', allowed_tags=['dlami'])
    cli_logger.newline()