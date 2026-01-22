from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def install_ports(module, port_path, ports, variant, stdout, stderr):
    """ Installs one or more ports if not already installed. """
    install_c = 0
    for port in ports:
        if query_port(module, port_path, port):
            continue
        rc, out, err = module.run_command('%s install %s %s' % (port_path, port, variant))
        stdout += out
        stderr += err
        if not query_port(module, port_path, port):
            module.fail_json(msg='Failed to install %s: %s' % (port, err), stdout=stdout, stderr=stderr)
        install_c += 1
    if install_c > 0:
        module.exit_json(changed=True, msg='Installed %s port(s)' % install_c, stdout=stdout, stderr=stderr)
    module.exit_json(changed=False, msg='Port(s) already present', stdout=stdout, stderr=stderr)