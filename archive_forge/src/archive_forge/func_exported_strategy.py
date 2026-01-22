from __future__ import absolute_import, division, print_function
import datetime
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.scaleway import (
def exported_strategy(module, account_api, backup):
    if backup is None:
        module.fail_json(msg='Backup "%s" not found' % module.params['id'])
    if backup['download_url'] is not None:
        module.exit_json(changed=False, metadata=backup)
    if module.check_mode:
        module.exit_json(changed=True)
    backup = wait_to_complete_state_transition(module, account_api, backup)
    response = account_api.post('/rdb/v1/regions/%s/backups/%s/export' % (module.params.get('region'), backup['id']), {})
    if response.ok:
        result = wait_to_complete_state_transition(module, account_api, response.json)
        module.exit_json(changed=True, metadata=result)
    module.fail_json(msg='Error exporting backup [{0}: {1}]'.format(response.status_code, response.json))