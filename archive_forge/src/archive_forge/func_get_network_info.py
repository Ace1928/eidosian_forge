from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
def get_network_info(module, executable, name):
    command = [executable, 'network', 'inspect']
    if not name:
        all_names = [executable, 'network', 'ls', '-q']
        rc, out, err = module.run_command(all_names)
        if rc != 0:
            module.fail_json(msg='Unable to get list of networks: %s' % err)
        name = out.split()
        if not name:
            return ([], out, err)
        command += name
    else:
        command.append(name)
    rc, out, err = module.run_command(command)
    if rc != 0 or 'unable to find network configuration' in err:
        module.fail_json(msg='Unable to gather info for %s: %s' % (name, err))
    if not out or json.loads(out) is None:
        return ([], out, err)
    return (json.loads(out), out, err)