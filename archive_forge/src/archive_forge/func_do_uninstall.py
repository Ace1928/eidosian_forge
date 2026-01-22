from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def do_uninstall(module, name, backend):
    atomic_bin = module.get_bin_path('atomic')
    args = [atomic_bin, 'uninstall', '--storage=%s' % backend, name]
    rc, out, err = module.run_command(args, check_rc=False)
    if rc != 0:
        module.fail_json(rc=rc, msg=err)
    module.exit_json(msg=out, changed=True)