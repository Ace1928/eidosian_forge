from __future__ import absolute_import, division, print_function
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def convert_to_kb(self, variable, converted_parameters):
    """
        Convert a number 10m in to its correct KB size
        :param variable: the Parameter we are going to covert
        :param converted_parameters: Dic of all parameters
        :return:
        """
    value = converted_parameters.get(variable)
    if len(value) < 2:
        self.module.fail_json(msg="%s must start with a number, and must end with a k, m, g or t, found '%s'." % (variable, value))
    if value[-1] not in ['k', 'm', 'g', 't']:
        self.module.fail_json(msg='%s must end with a k, m, g or t, found %s in %s.' % (variable, value[-1], value))
    try:
        digits = int(value[:-1])
    except ValueError:
        self.module.fail_json(msg='%s must start with a number, found %s in %s.' % (variable, value[:-1], value))
    return self._size_unit_map[value[-1]] * digits