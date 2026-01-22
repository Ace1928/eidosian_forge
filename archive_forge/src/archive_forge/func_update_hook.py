from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def update_hook(repo, hook, module):
    config = _create_hook_config(module)
    try:
        hook.update()
        hook.edit(name='web', config=config, events=module.params['events'], active=module.params['active'])
        changed = hook.update()
    except github.GithubException as err:
        module.fail_json(msg='Unable to modify hook for repository %s: %s' % (repo.full_name, to_native(err)))
    data = {'hook_id': hook.id}
    return (changed, data)