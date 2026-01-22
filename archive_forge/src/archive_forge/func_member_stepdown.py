from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def member_stepdown(client, module):
    """
    client - MongoDB Client
    module - Ansible module object
    """
    try:
        from collections import OrderedDict
    except ImportError as excep:
        try:
            from ordereddict import OrderedDict
        except ImportError as excep:
            module.fail_json(msg='Cannot import OrderedDict class. You can probably install with: pip install ordereddict: %s' % to_native(excep))
    iterations = 0
    failures = 0
    poll = module.params['poll']
    interval = module.params['interval']
    stepdown_seconds = module.params['stepdown_seconds']
    secondary_catch_up = module.params['secondary_catch_up']
    force = module.params['force']
    return_doc = {}
    status = None
    while iterations < poll:
        try:
            iterations += 1
            return_doc['iterations'] = iterations
            myStateStr = member_status(client)
            if myStateStr == 'PRIMARY':
                if module.check_mode:
                    return_doc['msg'] = 'member was stepped down'
                    return_doc['changed'] = True
                    status = True
                    break
                else:
                    cmd_doc = OrderedDict([('replSetStepDown', stepdown_seconds), ('secondaryCatchUpPeriodSecs', secondary_catch_up), ('force', force)])
                    try:
                        client.admin.command(cmd_doc)
                    except Exception as excep:
                        if str(excep) == 'connection closed':
                            pass
                        else:
                            raise excep
                    return_doc['changed'] = True
                    status = True
                    return_doc['msg'] = 'member was stepped down'
                    break
            elif myStateStr in ['SECONDARY', 'ARBITER']:
                return_doc['msg'] = 'member was already at {0} state'.format(myStateStr)
                return_doc['changed'] = False
                status = True
                break
            elif myStateStr in ['STARTUP', 'RECOVERING', 'STARTUP2', 'ROLLBACK']:
                time.sleep(interval)
            else:
                return_doc['msg'] = 'Unexpected member state {0}'.format(myStateStr)
                return_doc['changed'] = False
                status = False
                break
        except Exception as e:
            failures += 1
            return_doc['failed'] = True
            return_doc['changed'] = False
            return_doc['msg'] = str(e)
            status = False
            if iterations == poll:
                break
            else:
                time.sleep(interval)
    return (status, return_doc['msg'], return_doc)