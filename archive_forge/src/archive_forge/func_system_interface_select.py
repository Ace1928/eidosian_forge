from __future__ import absolute_import, division, print_function
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.argspec.system.system import SystemArgs
def system_interface_select(self):
    fos = self._fos
    vdom = self._module.params['vdom']
    query_string = '?vdom=' + vdom
    system_interface_select_param = self._subset['filters']
    if system_interface_select_param:
        for filter in system_interface_select_param:
            for key, val in filter.items():
                if val:
                    query_string += '&' + str(key) + '=' + str(val)
    return fos.monitor('system', self._subset['fact'][len('system_'):].replace('_', '/') + query_string, vdom=None)