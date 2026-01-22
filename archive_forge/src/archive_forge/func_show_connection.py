from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
import re
def show_connection(self):
    cmd = [self.nmcli_bin, '--show-secrets', 'con', 'show', self.conn_name]
    rc, out, err = self.execute_command(cmd)
    if rc != 0:
        raise NmcliModuleError(err)
    p_enum_value = re.compile('^([-]?\\d+) \\((\\w+)\\)$')
    conn_info = dict()
    for line in out.splitlines():
        pair = line.split(':', 1)
        key = pair[0].strip()
        key_type = self.settings_type(key)
        if key and len(pair) > 1:
            raw_value = pair[1].lstrip()
            if raw_value == '--':
                if key_type == list:
                    conn_info[key] = []
                else:
                    conn_info[key] = None
            elif key == 'bond.options':
                opts = raw_value.split(',')
                for opt in opts:
                    alias_pair = opt.split('=', 1)
                    if len(alias_pair) > 1:
                        alias_key = alias_pair[0]
                        alias_value = alias_pair[1]
                        conn_info[alias_key] = alias_value
            elif key in ('ipv4.routes', 'ipv6.routes'):
                conn_info[key] = [s.strip() for s in raw_value.split(';')]
            elif key_type == list:
                conn_info[key] = [s.strip() for s in raw_value.split(',')]
            else:
                m_enum = p_enum_value.match(raw_value)
                if m_enum is not None:
                    value = m_enum.group(1)
                else:
                    value = raw_value
                conn_info[key] = value
    return conn_info