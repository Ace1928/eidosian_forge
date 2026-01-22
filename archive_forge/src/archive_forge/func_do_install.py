from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def do_install(module, mode, rootfs, container, image, values_list, backend):
    system_list = ['--system'] if mode == 'system' else []
    user_list = ['--user'] if mode == 'user' else []
    rootfs_list = ['--rootfs=%s' % rootfs] if rootfs else []
    atomic_bin = module.get_bin_path('atomic')
    args = [atomic_bin, 'install', '--storage=%s' % backend, '--name=%s' % container] + system_list + user_list + rootfs_list + values_list + [image]
    rc, out, err = module.run_command(args, check_rc=False)
    if rc != 0:
        module.fail_json(rc=rc, msg=err)
    else:
        changed = 'Extracting' in out or 'Copying blob' in out
        module.exit_json(msg=out, changed=changed)