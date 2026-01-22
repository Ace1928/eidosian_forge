from __future__ import (absolute_import, division, print_function)
import copy
import json
import os
import re
import traceback
from io import BytesIO
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, json_dict_bytes_to_unicode, missing_required_lib
from ansible.module_utils.six import iteritems, string_types
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.common._collections_compat import MutableMapping
def set_target_inner(module, tree, xpath, namespaces, attribute, value):
    changed = False
    try:
        if not is_node(tree, xpath, namespaces):
            changed = check_or_make_target(module, tree, xpath, namespaces)
    except Exception as e:
        missing_namespace = ''
        if tree.getroot().nsmap and ':' not in xpath:
            missing_namespace = 'XML document has namespace(s) defined, but no namespace prefix(es) used in xpath!\n'
        module.fail_json(msg='%sXpath %s causes a failure: %s\n  -- tree is %s' % (missing_namespace, xpath, e, etree.tostring(tree, pretty_print=True)), exception=traceback.format_exc())
    if not is_node(tree, xpath, namespaces):
        module.fail_json(msg='Xpath %s does not reference a node! tree is %s' % (xpath, etree.tostring(tree, pretty_print=True)))
    for element in tree.xpath(xpath, namespaces=namespaces):
        if not attribute:
            changed = changed or element.text != value
            if element.text != value:
                element.text = value
        else:
            changed = changed or element.get(attribute) != value
            if ':' in attribute:
                attr_ns, attr_name = attribute.split(':')
                attribute = '{{{0}}}{1}'.format(namespaces[attr_ns], attr_name)
            if element.get(attribute) != value:
                element.set(attribute, value)
    return changed