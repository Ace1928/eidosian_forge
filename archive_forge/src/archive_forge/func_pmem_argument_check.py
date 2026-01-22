from __future__ import (absolute_import, division, print_function)
import json
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, human_to_bytes
def pmem_argument_check(self):

    def namespace_check(self):
        command = ['list', '-R']
        out = self.pmem_run_ndctl(command)
        if not out:
            return 'Available region(s) is not in this system.'
        region = json.loads(out)
        aligns = self.pmem_get_region_align_size(region)
        if len(aligns) != 1:
            return 'Not supported the regions whose alignment size is different.'
        available_size = self.pmem_get_available_region_size(region)
        types = self.pmem_get_available_region_type(region)
        for ns in self.namespace:
            if ns['size']:
                try:
                    size_byte = human_to_bytes(ns['size'])
                except ValueError:
                    return 'The format of size: NNN TB|GB|MB|KB|T|G|M|K|B'
                if size_byte % aligns[0] != 0:
                    return 'size: %s should be align with %d' % (ns['size'], aligns[0])
                is_space_enough = False
                for i, avail in enumerate(available_size):
                    if avail > size_byte:
                        available_size[i] -= size_byte
                        is_space_enough = True
                        break
                if is_space_enough is False:
                    return 'There is not available region for size: %s' % ns['size']
                ns['size_byte'] = size_byte
            elif len(self.namespace) != 1:
                return 'size option is required to configure multiple namespaces'
            if ns['type'] not in types:
                return 'type %s is not supported in this system. Supported type: %s' % (ns['type'], types)
        return None

    def percent_check(self, appdirect, memmode, reserved=None):
        if appdirect is None or (appdirect < 0 or appdirect > 100):
            return 'appdirect percent should be from 0 to 100.'
        if memmode is None or (memmode < 0 or memmode > 100):
            return 'memorymode percent should be from 0 to 100.'
        if reserved is None:
            if appdirect + memmode > 100:
                return 'Total percent should be less equal 100.'
        else:
            if reserved < 0 or reserved > 100:
                return 'reserved percent should be from 0 to 100.'
            if appdirect + memmode + reserved != 100:
                return 'Total percent should be 100.'

    def socket_id_check(self):
        command = ['show', '-o', 'nvmxml', '-socket']
        out = self.pmem_run_ipmctl(command)
        sockets_dict = xmltodict.parse(out, dict_constructor=dict)['SocketList']['Socket']
        socket_ids = []
        for sl in sockets_dict:
            socket_ids.append(int(sl['SocketID'], 16))
        for skt in self.socket:
            if skt['id'] not in socket_ids:
                return 'Invalid socket number: %d' % skt['id']
        return None
    if self.namespace:
        return namespace_check(self)
    elif self.socket is None:
        return percent_check(self, self.appdirect, self.memmode, self.reserved)
    else:
        ret = socket_id_check(self)
        if ret is not None:
            return ret
        for skt in self.socket:
            ret = percent_check(self, skt['appdirect'], skt['memorymode'], skt['reserved'])
            if ret is not None:
                return ret
        return None