from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def state_vg(module, vg, state, vg_validation):
    vg_state, msg = vg_validation
    if vg_state is None:
        module.fail_json(msg=msg)
    if state == 'varyon':
        if vg_state is True:
            changed = False
            return (changed, msg)
        changed = True
        msg = ''
        if not module.check_mode:
            varyonvg_cmd = module.get_bin_path('varyonvg', True)
            rc, varyonvg_out, err = module.run_command([varyonvg_cmd, vg])
            if rc != 0:
                module.fail_json(msg="Command 'varyonvg' failed.", rc=rc, err=err)
        msg = 'Varyon volume group %s completed.' % vg
        return (changed, msg)
    elif state == 'varyoff':
        if vg_state is False:
            changed = False
            return (changed, msg)
        changed = True
        msg = ''
        if not module.check_mode:
            varyonvg_cmd = module.get_bin_path('varyoffvg', True)
            rc, varyonvg_out, stderr = module.run_command([varyonvg_cmd, vg])
            if rc != 0:
                module.fail_json(msg="Command 'varyoffvg' failed.", rc=rc, stdout=varyonvg_out, stderr=stderr)
        msg = 'Varyoff volume group %s completed.' % vg
        return (changed, msg)