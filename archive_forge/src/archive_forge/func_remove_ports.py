from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def remove_ports(module, port_path, ports, stdout, stderr):
    """ Uninstalls one or more ports if installed. """
    remove_c = 0
    for port in ports:
        if not query_port(module, port_path, port):
            continue
        rc, out, err = module.run_command('%s uninstall %s' % (port_path, port))
        stdout += out
        stderr += err
        if query_port(module, port_path, port):
            module.fail_json(msg='Failed to remove %s: %s' % (port, err), stdout=stdout, stderr=stderr)
        remove_c += 1
    if remove_c > 0:
        module.exit_json(changed=True, msg='Removed %s port(s)' % remove_c, stdout=stdout, stderr=stderr)
    module.exit_json(changed=False, msg='Port(s) already absent', stdout=stdout, stderr=stderr)