from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def replicaset_reconfigure(module, client, config, force, max_time_ms):
    config['version'] += 1
    try:
        from collections import OrderedDict
    except ImportError as excep:
        try:
            from ordereddict import OrderedDict
        except ImportError as excep:
            module.fail_json(msg='Cannot import OrderedDict class. You can probably install with: pip install ordereddict: %s' % to_native(excep))
    cmd_doc = OrderedDict([('replSetReconfig', config), ('force', force)])
    if max_time_ms is not None:
        cmd_doc.update({'maxTimeMS': max_time_ms})
    client.admin.command(cmd_doc)