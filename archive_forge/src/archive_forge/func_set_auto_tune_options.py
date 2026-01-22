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
def set_auto_tune_options(module, current_domain_config, desired_domain_config, change_set):
    changed = False
    auto_tune_config = desired_domain_config['AutoTuneOptions']
    auto_tune_opts = module.params.get('auto_tune_options')
    if auto_tune_opts is None:
        return changed
    schedules = auto_tune_opts.get('maintenance_schedules')
    if auto_tune_opts.get('desired_state') is not None:
        auto_tune_config['DesiredState'] = auto_tune_opts.get('desired_state')
    if auto_tune_config['DesiredState'] != 'ENABLED':
        desired_domain_config['AutoTuneOptions'] = {'DesiredState': 'DISABLED'}
    elif schedules is not None:
        auto_tune_config['MaintenanceSchedules'] = []
        for s in schedules:
            schedule_entry = {}
            start_at = s.get('start_at')
            if start_at is not None:
                if isinstance(start_at, datetime.datetime):
                    start_at = start_at.strftime('%Y-%m-%d')
                schedule_entry['StartAt'] = start_at
            duration_opt = s.get('duration')
            if duration_opt is not None:
                schedule_entry['Duration'] = {}
                if duration_opt.get('value') is not None:
                    schedule_entry['Duration']['Value'] = duration_opt.get('value')
                if duration_opt.get('unit') is not None:
                    schedule_entry['Duration']['Unit'] = duration_opt.get('unit')
            if s.get('cron_expression_for_recurrence') is not None:
                schedule_entry['CronExpressionForRecurrence'] = s.get('cron_expression_for_recurrence')
            auto_tune_config['MaintenanceSchedules'].append(schedule_entry)
    if current_domain_config is not None:
        if current_domain_config['AutoTuneOptions']['DesiredState'] != auto_tune_config['DesiredState']:
            change_set.append(f'AutoTuneOptions.DesiredState changed from {current_domain_config['AutoTuneOptions']['DesiredState']} to {auto_tune_config['DesiredState']}')
            changed = True
        if auto_tune_config['MaintenanceSchedules'] != current_domain_config['AutoTuneOptions']['MaintenanceSchedules']:
            change_set.append(f'AutoTuneOptions.MaintenanceSchedules changed from {current_domain_config['AutoTuneOptions']['MaintenanceSchedules']} to {auto_tune_config['MaintenanceSchedules']}')
            changed = True
    return changed