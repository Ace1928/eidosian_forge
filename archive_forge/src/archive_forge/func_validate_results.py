from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.vyos import run_commands
def validate_results(module, loss, results):
    state = module.params['state']
    if state == 'present' and int(loss) == 100:
        module.fail_json(msg='Ping failed unexpectedly', **results)
    elif state == 'absent' and int(loss) < 100:
        module.fail_json(msg='Ping succeeded unexpectedly', **results)