from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def post_config(self, parent_class=None):
    """
        This method is used to handle the logic when the modules state is equal to present. The method only pushes a change if
        the object has differences than what exists on the APIC, and if check_mode is False. A successful change will mark the
        module as changed.
        """
    if not self.config:
        return
    elif not self.module.check_mode:
        url = self.url
        if parent_class is not None:
            if self.params.get('port') is not None:
                url = '{protocol}://{host}:{port}/{path}'.format(path=self.parent_path, **self.module.params)
            else:
                url = '{protocol}://{host}/{path}'.format(path=self.parent_path, **self.module.params)
            self.config = {parent_class: {'attributes': {}, 'children': [self.config]}}
        self.api_call('POST', url, json.dumps(self.config), return_response=False)
    else:
        self.result['changed'] = True
        self.method = 'POST'