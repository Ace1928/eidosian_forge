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
def delete_xpath_target(module, tree, xpath, namespaces):
    """ Delete an attribute or element from a tree """
    changed = False
    try:
        for result in tree.xpath(xpath, namespaces=namespaces):
            changed = True
            if is_attribute(tree, xpath, namespaces):
                parent = result.getparent()
                parent.attrib.pop(result.attrname)
            elif is_node(tree, xpath, namespaces):
                result.getparent().remove(result)
            else:
                raise Exception('Impossible error')
    except Exception as e:
        module.fail_json(msg="Couldn't delete xpath target: %s (%s)" % (xpath, e))
    else:
        finish(module, tree, xpath, namespaces, changed=changed)