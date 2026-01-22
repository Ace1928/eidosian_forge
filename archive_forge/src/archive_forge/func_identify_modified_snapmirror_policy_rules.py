from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def identify_modified_snapmirror_policy_rules(self, current=None):
    """
        Identify self.parameters rules that will be modified or not.
        :return: List of 'modified' rules and a list of 'unmodified' rules
                 e.g. [{'snapmirror_label': 'daily', 'keep': 7, 'prefix': '', 'schedule': ''}, ... ]
        """
    modified_rules = []
    unmodified_rules = []
    if 'snapmirror_label' in self.parameters:
        for snapmirror_label in self.parameters['snapmirror_label']:
            snapmirror_label = snapmirror_label.strip()
            if current is not None and 'snapmirror_label' in current and (snapmirror_label in current['snapmirror_label']):
                modified = False
                rule = {'snapmirror_label': snapmirror_label}
                current_snapmirror_label_index = current['snapmirror_label'].index(snapmirror_label)
                snapmirror_label_index = self.parameters['snapmirror_label'].index(snapmirror_label)
                if self.set_rule(rule, 'keep', current, snapmirror_label_index, current_snapmirror_label_index):
                    modified = True
                if self.set_rule(rule, 'prefix', current, snapmirror_label_index, current_snapmirror_label_index):
                    modified = True
                if self.set_rule(rule, 'schedule', current, snapmirror_label_index, current_snapmirror_label_index):
                    modified = True
                if modified:
                    modified_rules.append(rule)
                else:
                    unmodified_rules.append(rule)
    return (modified_rules, unmodified_rules)