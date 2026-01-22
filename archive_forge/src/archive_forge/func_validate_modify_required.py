from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def validate_modify_required(self, modify, current):
    if len(modify['policy']['statements']) != len(current['policy']['statements']):
        return True
    match_found = []
    for statement in modify['policy']['statements']:
        for index, current_statement in enumerate(current['policy']['statements']):
            if index in match_found:
                continue
            statement_modified = self.na_helper.get_modified_attributes(current_statement, statement)
            if not statement_modified:
                match_found.append(index)
                break
            if len(statement_modified) > 1:
                continue
            if statement_modified.get('conditions'):
                if not current_statement['conditions']:
                    continue
                if len(statement_modified.get('conditions')) != len(current_statement['conditions']):
                    continue

                def require_modify(desired, current):
                    for condition in desired:
                        if condition.get('operator'):
                            for current_condition in current:
                                if condition['operator'] == current_condition['operator']:
                                    condition_modified = self.na_helper.get_modified_attributes(current_condition, condition)
                                    if condition_modified:
                                        return True
                        else:
                            return True
                if not require_modify(statement_modified['conditions'], current_statement['conditions']):
                    match_found.append(index)
                    break
    return not match_found or len(match_found) != len(modify['policy']['statements'])