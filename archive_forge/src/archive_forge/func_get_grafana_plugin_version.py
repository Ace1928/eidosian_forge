from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def get_grafana_plugin_version(module, params):
    """
    Fetch grafana installed plugin version. Return None if plugin is not installed.

    :param module: ansible module object. used to run system commands.
    :param params: ansible module params.
    """
    grafana_cli = grafana_cli_bin(params)
    rc, stdout, stderr = module.run_command('{0} ls'.format(grafana_cli))
    stdout_lines = stdout.split('\n')
    for line in stdout_lines:
        if line.find(' @ ') != -1:
            line = line.rstrip()
            plugin_name, plugin_version = parse_version(line)
            if plugin_name == params['name']:
                return plugin_version
    return None