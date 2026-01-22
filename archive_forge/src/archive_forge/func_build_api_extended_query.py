from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.quoting import (
from ansible_collections.community.routeros.plugins.module_utils.api import (
import re
def build_api_extended_query(self, item):
    if item['attribute'] not in self.extended_query['attributes']:
        self.errors("'%s' attribute is not in attributes: %s" % (item, self.extended_query['attributes']))
    if item['is'] in ('eq', '=='):
        return self.query_keys[item['attribute']] == item['value']
    elif item['is'] in ('not', '!='):
        return self.query_keys[item['attribute']] != item['value']
    elif item['is'] in ('less', '<'):
        return self.query_keys[item['attribute']] < item['value']
    elif item['is'] in ('more', '>'):
        return self.query_keys[item['attribute']] > item['value']
    elif item['is'] == 'in':
        return self.query_keys[item['attribute']].In(*item['value'])
    else:
        self.errors("'%s' is not operator for 'is'" % item['is'])