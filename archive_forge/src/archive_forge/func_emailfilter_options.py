from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.data_post_processor import (
def emailfilter_options(data, fos):
    vdom = data['vdom']
    emailfilter_options_data = data['emailfilter_options']
    filtered_data = underscore_to_hyphen(filter_emailfilter_options_data(emailfilter_options_data))
    return fos.set('emailfilter', 'options', data=filtered_data, vdom=vdom)