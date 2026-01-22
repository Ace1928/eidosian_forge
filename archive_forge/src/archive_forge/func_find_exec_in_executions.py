from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.identity.keycloak.keycloak \
from ansible.module_utils.basic import AnsibleModule
def find_exec_in_executions(searched_exec, executions):
    """
    Search if exec is contained in the executions.
    :param searched_exec: Execution to search for.
    :param executions: List of executions.
    :return: Index of the execution, -1 if not found..
    """
    for i, existing_exec in enumerate(executions, start=0):
        if 'providerId' in existing_exec and 'providerId' in searched_exec and (existing_exec['providerId'] == searched_exec['providerId']) or ('displayName' in existing_exec and 'displayName' in searched_exec and (existing_exec['displayName'] == searched_exec['displayName'])):
            return i
    return -1