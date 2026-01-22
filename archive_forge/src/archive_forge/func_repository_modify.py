from __future__ import absolute_import, division, print_function
import os
from fnmatch import fnmatch
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
def repository_modify(module, rhsm, state, name, purge=False):
    name = set(name)
    current_repo_list = rhsm.list_repositories()
    updated_repo_list = deepcopy(current_repo_list)
    matched_existing_repo = {}
    for repoid in name:
        matched_existing_repo[repoid] = []
        for idx, repo in enumerate(current_repo_list):
            if fnmatch(repo['id'], repoid):
                matched_existing_repo[repoid].append(repo)
                updated_repo_list[idx]['enabled'] = True if state == 'enabled' else False
    changed = False
    results = []
    diff_before = ''
    diff_after = ''
    rhsm_arguments = []
    for repoid in matched_existing_repo:
        if len(matched_existing_repo[repoid]) == 0:
            results.append('%s is not a valid repository ID' % repoid)
            module.fail_json(results=results, msg='%s is not a valid repository ID' % repoid)
        for repo in matched_existing_repo[repoid]:
            if state in ['disabled', 'absent']:
                if repo['enabled']:
                    changed = True
                    diff_before += "Repository '%s' is enabled for this system\n" % repo['id']
                    diff_after += "Repository '%s' is disabled for this system\n" % repo['id']
                results.append("Repository '%s' is disabled for this system" % repo['id'])
                rhsm_arguments += ['--disable', repo['id']]
            elif state in ['enabled', 'present']:
                if not repo['enabled']:
                    changed = True
                    diff_before += "Repository '%s' is disabled for this system\n" % repo['id']
                    diff_after += "Repository '%s' is enabled for this system\n" % repo['id']
                results.append("Repository '%s' is enabled for this system" % repo['id'])
                rhsm_arguments += ['--enable', repo['id']]
    if purge:
        enabled_repo_ids = set((repo['id'] for repo in updated_repo_list if repo['enabled']))
        matched_repoids_set = set(matched_existing_repo.keys())
        difference = enabled_repo_ids.difference(matched_repoids_set)
        if len(difference) > 0:
            for repoid in difference:
                changed = True
                diff_before.join("Repository '{repoid}'' is enabled for this system\n".format(repoid=repoid))
                diff_after.join("Repository '{repoid}' is disabled for this system\n".format(repoid=repoid))
                results.append("Repository '{repoid}' is disabled for this system".format(repoid=repoid))
                rhsm_arguments.extend(['--disable', repoid])
            for updated_repo in updated_repo_list:
                if updated_repo['id'] in difference:
                    updated_repo['enabled'] = False
    diff = {'before': diff_before, 'after': diff_after, 'before_header': 'RHSM repositories', 'after_header': 'RHSM repositories'}
    if not module.check_mode and changed:
        rc, out, err = rhsm.run_repos(rhsm_arguments)
        results = out.splitlines()
    module.exit_json(results=results, changed=changed, repositories=updated_repo_list, diff=diff)