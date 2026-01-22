from __future__ import absolute_import, division, print_function
import json
import os
import re
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def validate_uuids(module):
    failed = [name for name, pvalue in [(x, module.params[x]) for x in ['uuid', 'image_uuid']] if pvalue and pvalue != '*' and (not is_valid_uuid(pvalue))]
    if failed:
        module.fail_json(msg='No valid UUID(s) found for: {0}'.format(', '.join(failed)))