from __future__ import absolute_import, division, print_function
import binascii
import socket
import struct
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native
def setup_host(self):
    if self.module.params['hostname'] is None or len(self.module.params['hostname']) == 0:
        self.module.fail_json(msg='name attribute could not be empty when adding or modifying host.')
    msg = None
    host_response = self.get_host(self.module.params['macaddr'])
    if host_response is None:
        msg = OmapiMessage.open(to_bytes('host', errors='surrogate_or_strict'))
        msg.message.append((to_bytes('create'), struct.pack('!I', 1)))
        msg.message.append((to_bytes('exclusive'), struct.pack('!I', 1)))
        msg.obj.append((to_bytes('hardware-address'), pack_mac(self.module.params['macaddr'])))
        msg.obj.append((to_bytes('hardware-type'), struct.pack('!I', 1)))
        msg.obj.append((to_bytes('name'), to_bytes(self.module.params['hostname'])))
        if self.module.params['ip'] is not None:
            msg.obj.append((to_bytes('ip-address', errors='surrogate_or_strict'), pack_ip(self.module.params['ip'])))
        stmt_join = ''
        if self.module.params['ddns']:
            stmt_join += 'ddns-hostname "{0}"; '.format(self.module.params['hostname'])
        try:
            if len(self.module.params['statements']) > 0:
                stmt_join += '; '.join(self.module.params['statements'])
                stmt_join += '; '
        except TypeError as e:
            self.module.fail_json(msg='Invalid statements found: %s' % to_native(e))
        if len(stmt_join) > 0:
            msg.obj.append((to_bytes('statements'), to_bytes(stmt_join)))
        try:
            response = self.omapi.query_server(msg)
            if response.opcode != OMAPI_OP_UPDATE:
                self.module.fail_json(msg='Failed to add host, ensure authentication and host parameters are valid.')
            self.module.exit_json(changed=True, lease=self.unpack_facts(response.obj))
        except OmapiError as e:
            self.module.fail_json(msg='OMAPI error: %s' % to_native(e))
    else:
        response_obj = self.unpack_facts(host_response.obj)
        fields_to_update = {}
        if to_bytes('ip-address', errors='surrogate_or_strict') not in response_obj or unpack_ip(response_obj[to_bytes('ip-address', errors='surrogate_or_strict')]) != self.module.params['ip']:
            fields_to_update['ip-address'] = pack_ip(self.module.params['ip'])
        if 'name' not in response_obj or response_obj['name'] != self.module.params['hostname']:
            self.module.fail_json(msg='Changing hostname is not supported. Old was %s, new is %s. Please delete host and add new.' % (response_obj['name'], self.module.params['hostname']))
        "\n            # It seems statements are not returned by OMAPI, then we cannot modify them at this moment.\n            if 'statements' not in response_obj and len(self.module.params['statements']) > 0 or                 response_obj['statements'] != self.module.params['statements']:\n                with open('/tmp/omapi', 'w') as fb:\n                    for (k,v) in iteritems(response_obj):\n                        fb.writelines('statements: %s %s\n' % (k, v))\n            "
        if len(fields_to_update) == 0:
            self.module.exit_json(changed=False, lease=response_obj)
        else:
            msg = OmapiMessage.update(host_response.handle)
            msg.update_object(fields_to_update)
        try:
            response = self.omapi.query_server(msg)
            if response.opcode != OMAPI_OP_STATUS:
                self.module.fail_json(msg='Failed to modify host, ensure authentication and host parameters are valid.')
            self.module.exit_json(changed=True)
        except OmapiError as e:
            self.module.fail_json(msg='OMAPI error: %s' % to_native(e))