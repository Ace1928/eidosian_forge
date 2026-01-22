from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.storage_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def server_probe(self, server_data):
    updates = []
    if self.ip and server_data['IP_address'] != self.ip:
        updates.append('ip')
    if self.facility is not None and server_data['facility'] != self.facility:
        updates.append('facility')
    if self.error and server_data['error'] != self.error:
        updates.append('error')
    if self.warning and server_data['warning'] != self.warning:
        updates.append('warning')
    if self.info and server_data['info'] != self.info:
        updates.append('info')
    if self.audit and server_data['audit'] != self.audit:
        updates.append('audit')
    if self.login and server_data['login'] != self.login:
        updates.append('login')
    if self.port is not None:
        if int(server_data['port']) != self.port:
            updates.append('port')
            updates.append('protocol')
    if self.protocol and server_data['protocol'] != self.protocol:
        updates.append('protocol')
    if self.cadf and server_data['cadf'] != self.cadf:
        updates.append('cadf')
    self.log('Syslogserver probe result: %s', updates)
    return updates