from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
def grafana_plugin(module, params):
    """
    Install update or remove grafana plugin

    :param module: ansible module object. used to run system commands.
    :param params: ansible module params.
    """
    grafana_cli = grafana_cli_bin(params)
    if params['state'] == 'present':
        grafana_plugin_version = get_grafana_plugin_version(module, params)
        if grafana_plugin_version is not None:
            if 'version' in params and params['version']:
                if params['version'] == grafana_plugin_version:
                    return {'msg': 'Grafana plugin already installed', 'changed': False, 'version': grafana_plugin_version}
                elif params['version'] == 'latest' or params['version'] is None:
                    latest_version = get_grafana_plugin_version_latest(module, params)
                    if latest_version == grafana_plugin_version:
                        return {'msg': 'Grafana plugin already installed', 'changed': False, 'version': grafana_plugin_version}
                    cmd = '{0} update {1}'.format(grafana_cli, params['name'])
                else:
                    cmd = '{0} install {1} {2}'.format(grafana_cli, params['name'], params['version'])
            else:
                return {'msg': 'Grafana plugin already installed', 'changed': False, 'version': grafana_plugin_version}
        elif 'version' in params:
            if params['version'] == 'latest' or params['version'] is None:
                cmd = '{0} install {1}'.format(grafana_cli, params['name'])
            else:
                cmd = '{0} install {1} {2}'.format(grafana_cli, params['name'], params['version'])
        else:
            cmd = '{0} install {1}'.format(grafana_cli, params['name'])
    else:
        cmd = '{0} uninstall {1}'.format(grafana_cli, params['name'])
    rc, stdout, stderr = module.run_command(cmd)
    if rc == 0:
        stdout_lines = stdout.split('\n')
        for line in stdout_lines:
            if line.find(params['name']):
                if line.find(' @ ') != -1:
                    line = line.rstrip()
                    plugin_name, plugin_version = parse_version(line)
                else:
                    plugin_version = None
                if params['state'] == 'present':
                    return {'msg': 'Grafana plugin {0} installed : {1}'.format(params['name'], cmd), 'changed': True, 'version': plugin_version}
                else:
                    return {'msg': 'Grafana plugin {0} uninstalled : {1}'.format(params['name'], cmd), 'changed': True}
    else:
        if params['state'] == 'absent' and stdout.find('plugin does not exist'):
            return {'msg': 'Grafana plugin {0} already uninstalled : {1}'.format(params['name'], cmd), 'changed': False}
        raise GrafanaCliException("'{0}' execution returned an error : [{1}] {2} {3}".format(cmd, rc, stdout, stderr))