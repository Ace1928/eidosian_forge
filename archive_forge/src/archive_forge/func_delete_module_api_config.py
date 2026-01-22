from __future__ import absolute_import, division, print_function
from ansible.module_utils.connection import Connection
from ansible.module_utils.six.moves.urllib.parse import quote_plus
from ansible.plugins.action import ActionBase
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.splunk.es.plugins.module_utils.splunk import (
from ansible_collections.splunk.es.plugins.modules.splunk_data_inputs_monitor import DOCUMENTATION
def delete_module_api_config(self, conn_request, config):
    before = []
    after = None
    changed = False
    for want_conf in config:
        search_by_name = self.search_for_resource_name(conn_request, want_conf['name'])
        if search_by_name:
            before.append(search_by_name)
            conn_request.delete_by_path('{0}/{1}'.format(self.api_object, quote_plus(want_conf['name'])))
            changed = True
            after = []
    res_config = {}
    res_config['after'] = after
    res_config['before'] = before
    return (res_config, changed)