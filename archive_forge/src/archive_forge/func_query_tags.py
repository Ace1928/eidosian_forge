from __future__ import absolute_import, division, print_function
import os
import sys
import time
import traceback
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import missing_required_lib, env_fallback
def query_tags(self, resource, resource_type):
    args = {'resourceid': resource['id'], 'resourcetype': resource_type}
    tags = self.query_api('listTags', **args)
    return self.get_tags(resource=tags, key='tag')