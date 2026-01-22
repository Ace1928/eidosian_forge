from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleActionFail
from ansible.module_utils.connection import Connection
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.plugins.action import ActionBase
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.splunk.es.plugins.module_utils.splunk import (
from ansible_collections.splunk.es.plugins.modules.splunk_correlation_searches import DOCUMENTATION
def map_objects_to_params(self, want_conf):
    res = {}
    res['action.correlationsearch.enabled'] = '1'
    res['is_scheduled'] = True
    res['dispatch.rt_backfill'] = True
    res['action.correlationsearch.label'] = want_conf['name']
    res.update(map_obj_to_params(want_conf, self.key_transform))
    if 'realtime_schedule' in res:
        if res['realtime_schedule'] == 'realtime':
            res['realtime_schedule'] = True
        else:
            res['realtime_schedule'] = False
    if 'alert.digest_mode' in res:
        if res['alert.digest_mode'] == 'once':
            res['alert.digest_mode'] = True
        else:
            res['alert.digest_mode'] = False
    if 'alert.suppress.fields' in res:
        res['alert.suppress.fields'] = ','.join(res['alert.suppress.fields'])
    if 'action.correlationsearch.annotations' in res and 'custom' in res['action.correlationsearch.annotations']:
        for ele in res['action.correlationsearch.annotations']['custom']:
            res['action.correlationsearch.annotations'][ele['framework']] = ele['custom_annotations']
        res['action.correlationsearch.annotations'].pop('custom')
        res['action.correlationsearch.annotations'] = json.dumps(res['action.correlationsearch.annotations'])
    return res