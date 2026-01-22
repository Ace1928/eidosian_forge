import datetime
import time
from copy import deepcopy
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.botocore import normalize_boto3_result
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def build_rule(client, module):
    name = module.params.get('name')
    abort_incomplete_multipart_upload_days = module.params.get('abort_incomplete_multipart_upload_days')
    expiration_date = parse_date(module.params.get('expiration_date'))
    expiration_days = module.params.get('expiration_days')
    expire_object_delete_marker = module.params.get('expire_object_delete_marker')
    noncurrent_version_expiration_days = module.params.get('noncurrent_version_expiration_days')
    noncurrent_version_transition_days = module.params.get('noncurrent_version_transition_days')
    noncurrent_version_transitions = module.params.get('noncurrent_version_transitions')
    noncurrent_version_storage_class = module.params.get('noncurrent_version_storage_class')
    noncurrent_version_keep_newer = module.params.get('noncurrent_version_keep_newer')
    prefix = module.params.get('prefix') or ''
    rule_id = module.params.get('rule_id')
    status = module.params.get('status')
    storage_class = module.params.get('storage_class')
    transition_date = parse_date(module.params.get('transition_date'))
    transition_days = module.params.get('transition_days')
    transitions = module.params.get('transitions')
    purge_transitions = module.params.get('purge_transitions')
    rule = dict(Filter=dict(Prefix=prefix), Status=status.title())
    if rule_id is not None:
        rule['ID'] = rule_id
    if abort_incomplete_multipart_upload_days:
        rule['AbortIncompleteMultipartUpload'] = {'DaysAfterInitiation': abort_incomplete_multipart_upload_days}
    if expiration_days is not None:
        rule['Expiration'] = dict(Days=expiration_days)
    elif expiration_date is not None:
        rule['Expiration'] = dict(Date=expiration_date.isoformat())
    elif expire_object_delete_marker is not None:
        rule['Expiration'] = dict(ExpiredObjectDeleteMarker=expire_object_delete_marker)
    if noncurrent_version_expiration_days or noncurrent_version_keep_newer:
        rule['NoncurrentVersionExpiration'] = dict()
    if noncurrent_version_expiration_days is not None:
        rule['NoncurrentVersionExpiration']['NoncurrentDays'] = noncurrent_version_expiration_days
    if noncurrent_version_keep_newer is not None:
        rule['NoncurrentVersionExpiration']['NewerNoncurrentVersions'] = noncurrent_version_keep_newer
    if transition_days is not None:
        rule['Transitions'] = [dict(Days=transition_days, StorageClass=storage_class.upper())]
    elif transition_date is not None:
        rule['Transitions'] = [dict(Date=transition_date.isoformat(), StorageClass=storage_class.upper())]
    if transitions is not None:
        if not rule.get('Transitions'):
            rule['Transitions'] = []
        for transition in transitions:
            t_out = dict()
            if transition.get('transition_date'):
                t_out['Date'] = transition['transition_date']
            elif transition.get('transition_days') is not None:
                t_out['Days'] = int(transition['transition_days'])
            if transition.get('storage_class'):
                t_out['StorageClass'] = transition['storage_class'].upper()
                rule['Transitions'].append(t_out)
    if noncurrent_version_transition_days is not None:
        rule['NoncurrentVersionTransitions'] = [dict(NoncurrentDays=noncurrent_version_transition_days, StorageClass=noncurrent_version_storage_class.upper())]
    if noncurrent_version_transitions is not None:
        if not rule.get('NoncurrentVersionTransitions'):
            rule['NoncurrentVersionTransitions'] = []
        for noncurrent_version_transition in noncurrent_version_transitions:
            t_out = dict()
            t_out['NoncurrentDays'] = noncurrent_version_transition['transition_days']
            if noncurrent_version_transition.get('storage_class'):
                t_out['StorageClass'] = noncurrent_version_transition['storage_class'].upper()
                rule['NoncurrentVersionTransitions'].append(t_out)
    return rule