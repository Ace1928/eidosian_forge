import datetime
import json
from copy import deepcopy
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
from ansible_collections.community.aws.plugins.module_utils.opensearch import compare_domain_versions
from ansible_collections.community.aws.plugins.module_utils.opensearch import ensure_tags
from ansible_collections.community.aws.plugins.module_utils.opensearch import get_domain_config
from ansible_collections.community.aws.plugins.module_utils.opensearch import get_domain_status
from ansible_collections.community.aws.plugins.module_utils.opensearch import get_target_increment_version
from ansible_collections.community.aws.plugins.module_utils.opensearch import normalize_opensearch
from ansible_collections.community.aws.plugins.module_utils.opensearch import parse_version
from ansible_collections.community.aws.plugins.module_utils.opensearch import wait_for_domain_status
def set_cluster_config(module, current_domain_config, desired_domain_config, change_set):
    changed = False
    cluster_config = desired_domain_config['ClusterConfig']
    cluster_opts = module.params.get('cluster_config')
    if cluster_opts is not None:
        if cluster_opts.get('instance_type') is not None:
            cluster_config['InstanceType'] = cluster_opts.get('instance_type')
        if cluster_opts.get('instance_count') is not None:
            cluster_config['InstanceCount'] = cluster_opts.get('instance_count')
        if cluster_opts.get('zone_awareness') is not None:
            cluster_config['ZoneAwarenessEnabled'] = cluster_opts.get('zone_awareness')
        if cluster_config['ZoneAwarenessEnabled']:
            if cluster_opts.get('availability_zone_count') is not None:
                cluster_config['ZoneAwarenessConfig'] = {'AvailabilityZoneCount': cluster_opts.get('availability_zone_count')}
        if cluster_opts.get('dedicated_master') is not None:
            cluster_config['DedicatedMasterEnabled'] = cluster_opts.get('dedicated_master')
        if cluster_config['DedicatedMasterEnabled']:
            if cluster_opts.get('dedicated_master_instance_type') is not None:
                cluster_config['DedicatedMasterType'] = cluster_opts.get('dedicated_master_instance_type')
            if cluster_opts.get('dedicated_master_instance_count') is not None:
                cluster_config['DedicatedMasterCount'] = cluster_opts.get('dedicated_master_instance_count')
        if cluster_opts.get('warm_enabled') is not None:
            cluster_config['WarmEnabled'] = cluster_opts.get('warm_enabled')
        if cluster_config['WarmEnabled']:
            if cluster_opts.get('warm_type') is not None:
                cluster_config['WarmType'] = cluster_opts.get('warm_type')
            if cluster_opts.get('warm_count') is not None:
                cluster_config['WarmCount'] = cluster_opts.get('warm_count')
    cold_storage_opts = None
    if cluster_opts is not None:
        cold_storage_opts = cluster_opts.get('cold_storage_options')
    if compare_domain_versions(desired_domain_config['EngineVersion'], 'Elasticsearch_7.9') < 0:
        if cold_storage_opts is not None and cold_storage_opts.get('enabled'):
            module.fail_json(msg='Cold Storage is not supported')
        cluster_config.pop('ColdStorageOptions', None)
        if current_domain_config is not None and 'ClusterConfig' in current_domain_config:
            current_domain_config['ClusterConfig'].pop('ColdStorageOptions', None)
    elif cold_storage_opts is not None and cold_storage_opts.get('enabled') is not None:
        cluster_config['ColdStorageOptions'] = {'Enabled': cold_storage_opts.get('enabled')}
    if current_domain_config is not None and current_domain_config['ClusterConfig'] != cluster_config:
        change_set.append(f'ClusterConfig changed from {current_domain_config['ClusterConfig']} to {cluster_config}')
        changed = True
    return changed