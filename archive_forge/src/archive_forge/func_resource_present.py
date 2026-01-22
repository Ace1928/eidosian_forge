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
def resource_present(self, resource, fact_name, create_method='create'):
    """
        Generic implementation of the present state for the OneView resources.

        It checks if the resource needs to be created or updated.

        :arg dict resource: Resource to create or update.
        :arg str fact_name: Name of the fact returned to the Ansible.
        :arg str create_method: Function of the OneView client that will be called for resource creation.
            Usually create or add.
        :return: A dictionary with the expected arguments for the AnsibleModule.exit_json
        """
    changed = False
    if 'newName' in self.data:
        self.data['name'] = self.data.pop('newName')
    if not resource:
        resource = getattr(self.resource_client, create_method)(self.data)
        msg = self.MSG_CREATED
        changed = True
    else:
        merged_data = resource.copy()
        merged_data.update(self.data)
        if self.compare(resource, merged_data):
            msg = self.MSG_ALREADY_PRESENT
        else:
            resource = self.resource_client.update(merged_data)
            changed = True
            msg = self.MSG_UPDATED
    return dict(msg=msg, changed=changed, ansible_facts={fact_name: resource})