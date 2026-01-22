from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
def compare_initiators(self, user_initiator, existing_initiator):
    """
        compare user input initiator with existing dict
        :return: True if matched, False otherwise
        """
    if user_initiator is None or existing_initiator is None:
        return False
    changed = False
    for param in user_initiator:
        if param == 'name':
            if user_initiator['name'] == existing_initiator['initiator_name']:
                pass
        elif param == 'initiator_id':
            pass
        elif user_initiator[param] == existing_initiator[param]:
            pass
        else:
            self.debug.append('Initiator: %s.  Changed: %s from: %s to %s' % (user_initiator['name'], param, str(existing_initiator[param]), str(user_initiator[param])))
            changed = True
    return changed