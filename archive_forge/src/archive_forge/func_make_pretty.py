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
def make_pretty(module, tree):
    xml_string = etree.tostring(tree, xml_declaration=True, encoding='UTF-8', pretty_print=module.params['pretty_print'])
    result = dict(changed=False)
    if module.params['path']:
        xml_file = module.params['path']
        with open(xml_file, 'rb') as xml_content:
            if xml_string != xml_content.read():
                result['changed'] = True
                if not module.check_mode:
                    if module.params['backup']:
                        result['backup_file'] = module.backup_local(module.params['path'])
                    tree.write(xml_file, xml_declaration=True, encoding='UTF-8', pretty_print=module.params['pretty_print'])
    elif module.params['xmlstring']:
        result['xmlstring'] = xml_string
        if xml_string != module.params['xmlstring']:
            result['changed'] = True
    module.exit_json(**result)