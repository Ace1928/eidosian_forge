from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def set_efficiency_attributes(self, options):
    for key, attr in self.sis_keys2zapi_set.items():
        value = self.parameters.get(key)
        if value is not None:
            if self.argument_spec[key]['type'] == 'bool':
                value = self.na_helper.get_value_for_bool(False, value)
            options[attr] = value
    if options.get('enable-inline-compression') == 'true' and 'enable-compression' not in options:
        options['enable-compression'] = 'true'