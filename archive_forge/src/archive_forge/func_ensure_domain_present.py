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
def ensure_domain_present(client, module):
    domain_name = module.params.get('domain_name')
    desired_domain_config = {'DomainName': module.params.get('domain_name'), 'EngineVersion': 'OpenSearch_1.1', 'ClusterConfig': {'InstanceType': 't2.small.search', 'InstanceCount': 2, 'ZoneAwarenessEnabled': False, 'DedicatedMasterEnabled': False, 'WarmEnabled': False}, 'EBSOptions': {'EBSEnabled': False}, 'EncryptionAtRestOptions': {'Enabled': False}, 'NodeToNodeEncryptionOptions': {'Enabled': False}, 'SnapshotOptions': {'AutomatedSnapshotStartHour': 0}, 'CognitoOptions': {'Enabled': False}, 'AdvancedSecurityOptions': {'Enabled': False}, 'DomainEndpointOptions': {'CustomEndpointEnabled': False}, 'AutoTuneOptions': {'DesiredState': 'DISABLED'}}
    current_domain_config, domain_arn = get_domain_config(client, module, domain_name)
    if current_domain_config is not None:
        desired_domain_config = deepcopy(current_domain_config)
    if module.params.get('engine_version') is not None:
        v = parse_version(module.params.get('engine_version'))
        if v is None:
            module.fail_json('Invalid engine_version. Must be Elasticsearch_X.Y or OpenSearch_X.Y')
        desired_domain_config['EngineVersion'] = module.params.get('engine_version')
    changed = False
    change_set = []
    changed |= set_cluster_config(module, current_domain_config, desired_domain_config, change_set)
    changed |= set_ebs_options(module, current_domain_config, desired_domain_config, change_set)
    changed |= set_encryption_at_rest_options(module, current_domain_config, desired_domain_config, change_set)
    changed |= set_node_to_node_encryption_options(module, current_domain_config, desired_domain_config, change_set)
    changed |= set_vpc_options(module, current_domain_config, desired_domain_config, change_set)
    changed |= set_snapshot_options(module, current_domain_config, desired_domain_config, change_set)
    changed |= set_cognito_options(module, current_domain_config, desired_domain_config, change_set)
    changed |= set_advanced_security_options(module, current_domain_config, desired_domain_config, change_set)
    changed |= set_domain_endpoint_options(module, current_domain_config, desired_domain_config, change_set)
    changed |= set_auto_tune_options(module, current_domain_config, desired_domain_config, change_set)
    changed |= set_access_policy(module, current_domain_config, desired_domain_config, change_set)
    if current_domain_config is not None:
        if desired_domain_config['EngineVersion'] != current_domain_config['EngineVersion']:
            changed = True
            change_set.append('EngineVersion changed')
            upgrade_domain(client, module, current_domain_config['EngineVersion'], desired_domain_config['EngineVersion'])
        if changed:
            if module.check_mode:
                module.exit_json(changed=True, msg=f'Would have updated domain if not in check mode: {change_set}')
            desired_domain_config.pop('EngineVersion', None)
            try:
                client.update_domain_config(**desired_domain_config)
            except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
                module.fail_json_aws(e, msg=f"Couldn't update domain {domain_name}")
    else:
        if module.params.get('access_policies') is None:
            module.fail_json('state is present but the following is missing: access_policies')
        changed = True
        if module.check_mode:
            module.exit_json(changed=True, msg='Would have created a domain if not in check mode')
        try:
            response = client.create_domain(**desired_domain_config)
            domain = response['DomainStatus']
            domain_arn = domain['ARN']
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            module.fail_json_aws(e, msg=f"Couldn't update domain {domain_name}")
    try:
        existing_tags = boto3_tag_list_to_ansible_dict(client.list_tags(ARN=domain_arn, aws_retry=True)['TagList'])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, f"Couldn't get tags for domain {domain_name}")
    desired_tags = module.params['tags']
    purge_tags = module.params['purge_tags']
    changed |= ensure_tags(client, module, domain_arn, existing_tags, desired_tags, purge_tags)
    if module.params.get('wait') and (not module.check_mode):
        wait_for_domain_status(client, module, domain_name, 'domain_available')
    domain = get_domain_status(client, module, domain_name)
    return dict(changed=changed, **normalize_opensearch(client, module, domain))