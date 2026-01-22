from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def semanage_fcontext_delete(module, result, target, ftype, setype, substitute, do_reload, sestore=''):
    """ Delete SELinux file context mapping definition from the policy. """
    changed = False
    prepared_diff = ''
    try:
        sefcontext = seobject.fcontextRecords(sestore)
        sefcontext.set_reload(do_reload)
        exists = semanage_fcontext_exists(sefcontext, target, ftype)
        substitute_exists = semanage_fcontext_substitute_exists(sefcontext, target)
        if exists and substitute is None:
            orig_seuser, orig_serole, orig_setype, orig_serange = exists
            if not module.check_mode:
                sefcontext.delete(target, ftype)
            changed = True
            if module._diff:
                prepared_diff += '# Deletion to semanage file context mappings\n'
                prepared_diff += '-%s      %s      %s:%s:%s:%s\n' % (target, ftype, exists[0], exists[1], exists[2], exists[3])
        if substitute_exists and setype is None and (substitute is not None and substitute_exists == substitute or substitute is None):
            orig_substitute = substitute_exists
            if not module.check_mode:
                sefcontext.delete(target, orig_substitute)
            changed = True
            if module._diff:
                prepared_diff += '# Deletion to semanage file context path substitutions\n'
                prepared_diff += '-%s = %s\n' % (target, orig_substitute)
    except Exception as e:
        module.fail_json(msg='%s: %s\n' % (e.__class__.__name__, to_native(e)))
    if module._diff and prepared_diff:
        result['diff'] = dict(prepared=prepared_diff)
    module.exit_json(changed=changed, **result)