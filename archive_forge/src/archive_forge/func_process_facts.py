from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.argspec.logging_global.logging_global import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.logging_global import (
def process_facts(self, objFinal):
    if objFinal:
        for ke, vl in iteritems(objFinal):
            if ke == 'files':
                _files = []
                for k, v in vl.items():
                    _files.append(v)
                objFinal[ke] = _files
                objFinal[ke] = sorted(objFinal[ke], key=lambda item: item['path'])
            elif ke == 'hosts':
                _hosts = []
                for k, v in vl.items():
                    _hosts.append(v)
                objFinal[ke] = _hosts
                objFinal[ke] = sorted(objFinal[ke], key=lambda item: item['hostname'])
            elif ke == 'users':
                _users = []
                for k, v in vl.items():
                    _users.append(v)
                objFinal[ke] = _users
                objFinal[ke] = sorted(objFinal[ke], key=lambda item: item['username'])
            elif ke == 'console' or ke == 'global_params':
                if objFinal[ke].get('facilities'):
                    objFinal[ke]['facilities'] = sorted(objFinal[ke]['facilities'], key=lambda item: item['facility'])
    return objFinal