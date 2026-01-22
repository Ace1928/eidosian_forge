from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def get_grafana_plugin_version_latest(module, params):
    """
    Fetch the latest version available from grafana-cli.
    Return the newest version number or None not found.

    :param module: ansible module object. used to run system commands.
    :param params: ansible module params.
    """
    grafana_cli = grafana_cli_bin(params)
    rc, stdout, stderr = module.run_command('{0} list-versions {1}'.format(grafana_cli, params['name']))
    stdout_lines = stdout.split('\n')
    if stdout_lines[0]:
        return stdout_lines[0].rstrip()
    return None