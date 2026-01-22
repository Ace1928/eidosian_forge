from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible.module_utils.connection import ConnectionError
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def get_delete_global_params(self, conf, match):
    requests = []
    url = 'data/openconfig-system:system/aaa/server-groups/server-group=RADIUS/config/'
    if conf.get('auth_type', None) and match.get('auth_type', None) and (match['auth_type'] != 'pap'):
        requests.append({'path': url + 'auth-type', 'method': DELETE})
    if conf.get('key', None) and match.get('key', None):
        requests.append({'path': url + 'secret-key', 'method': DELETE})
    if conf.get('timeout', None) and match.get('timeout', None) and (match['timeout'] != 5):
        requests.append({'path': url + 'timeout', 'method': DELETE})
    return requests