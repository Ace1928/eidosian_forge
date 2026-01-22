from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def tmplt_params(config_data):

    def templt_common(val, tmplt):
        if val.get('facility'):
            tmplt += ' facility {facility}'.format(facility=val['facility'])
        if val.get('severity'):
            tmplt += ' level {level}'.format(level=val['severity'])
        if val.get('protocol'):
            tmplt += ' protocol {protocol}'.format(protocol=val['protocol'])
        return tmplt
    tmplt = ''
    if config_data.get('global_params'):
        val = config_data.get('global_params')
        if not val.get('archive'):
            tmplt += 'system syslog global'
        tmplt = templt_common(val.get('facilities'), tmplt)
    elif config_data.get('console'):
        val = config_data.get('console')
        tmplt += 'system syslog console'
        tmplt = templt_common(val.get('facilities'), tmplt)
    elif config_data.get('users'):
        val = config_data.get('users')
        if val.get('username') and (not val.get('archive')):
            tmplt += 'system syslog user {username}'.format(username=val['username'])
        if val.get('facilities'):
            tmplt = templt_common(val.get('facilities'), tmplt)
    elif config_data.get('hosts'):
        val = config_data.get('hosts')
        if val.get('hostname') and (not val.get('archive')) and (not val.get('port')):
            tmplt += 'system syslog host {hostname}'.format(hostname=val['hostname'])
        if val.get('facilities'):
            tmplt = templt_common(val.get('facilities'), tmplt)
    elif config_data.get('files'):
        val = config_data.get('files')
        if val.get('path') and (not val.get('archive')):
            tmplt += 'system syslog file {path}'.format(path=val['path'])
        if val.get('facilities'):
            tmplt = templt_common(val.get('facilities'), tmplt)
    return tmplt