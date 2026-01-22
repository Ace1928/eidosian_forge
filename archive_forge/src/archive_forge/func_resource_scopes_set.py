from __future__ import (absolute_import, division, print_function)
import abc
import collections
import json
import os  # noqa: F401, pylint: disable=unused-import
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common._collections_compat import Mapping
def resource_scopes_set(self, state, fact_name, scope_uris):
    """
        Generic implementation of the scopes update PATCH for the OneView resources.
        It checks if the resource needs to be updated with the current scopes.
        This method is meant to be run after ensuring the present state.
        :arg dict state: Dict containing the data from the last state results in the resource.
            It needs to have the 'msg', 'changed', and 'ansible_facts' entries.
        :arg str fact_name: Name of the fact returned to the Ansible.
        :arg list scope_uris: List with all the scope URIs to be added to the resource.
        :return: A dictionary with the expected arguments for the AnsibleModule.exit_json
        """
    if scope_uris is None:
        scope_uris = []
    resource = state['ansible_facts'][fact_name]
    operation_data = dict(operation='replace', path='/scopeUris', value=scope_uris)
    if resource['scopeUris'] is None or set(resource['scopeUris']) != set(scope_uris):
        state['ansible_facts'][fact_name] = self.resource_client.patch(resource['uri'], **operation_data)
        state['changed'] = True
        state['msg'] = self.MSG_UPDATED
    return state