from __future__ import absolute_import, division, print_function
import asyncio
import os
import urllib
from ansible.module_utils._text import to_native
from ansible.errors import AnsibleLookupError
from ansible_collections.cloud.common.plugins.module_utils.turbo.exceptions import (
from ansible_collections.vmware.vmware_rest.plugins.module_utils.vmware_rest import (
@staticmethod
def ensure_result(result, object_type, object_name=None):
    object_name_decoded = None
    if object_name:
        object_name_decoded = urllib.parse.unquote(object_name)
    if not result or (object_name_decoded and object_name_decoded not in result[0].values()):
        return ''

    def _filter_result(result):
        return [obj for obj in result if '%2f' not in obj['name']]
    result = _filter_result(result)
    if result and len(result) > 1:
        raise AnsibleLookupError('More than one object available: [%s].' % ', '.join(list((f'{item['name']} => {item[object_type]}' for item in result))))
    try:
        object_moid = result[0][object_type]
    except (TypeError, KeyError, IndexError) as e:
        raise AnsibleLookupError(to_native(e))
    return object_moid