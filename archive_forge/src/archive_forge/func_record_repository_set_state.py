from __future__ import absolute_import, division, print_function
from ansible_collections.theforeman.foreman.plugins.module_utils.foreman_helper import KatelloEntityAnsibleModule
def record_repository_set_state(module, record_data, repo, state_before, state_after):
    repo_change_data = record_data.copy()
    repo_change_data['repo_name'] = repo
    repo_change_data['state'] = state_before
    repo_change_data_after = repo_change_data.copy()
    repo_change_data_after['state'] = state_after
    module.record_before('repository_sets', repo_change_data)
    module.record_after('repository_sets', repo_change_data_after)
    module.record_after_full('repository_sets', repo_change_data_after)