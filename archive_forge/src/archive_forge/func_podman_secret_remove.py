from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import get_podman_version
def podman_secret_remove(module, executable, name):
    changed = False
    rc, out, err = module.run_command([executable, 'secret', 'rm', name])
    if rc == 0:
        changed = True
    elif 'no such secret' in err:
        pass
    else:
        module.fail_json(msg='Unable to remove secret: %s' % err)
    return {'changed': changed}