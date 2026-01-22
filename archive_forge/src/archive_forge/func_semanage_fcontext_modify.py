from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def semanage_fcontext_modify(module, result, target, ftype, setype, substitute, do_reload, serange, seuser, sestore=''):
    """ Add or modify SELinux file context mapping definition to the policy. """
    changed = False
    prepared_diff = ''
    try:
        sefcontext = seobject.fcontextRecords(sestore)
        sefcontext.set_reload(do_reload)
        if substitute is None:
            exists = semanage_fcontext_exists(sefcontext, target, ftype)
            if exists:
                orig_seuser, orig_serole, orig_setype, orig_serange = exists
                if seuser is None:
                    seuser = orig_seuser
                if serange is None:
                    serange = orig_serange
                if setype != orig_setype or seuser != orig_seuser or serange != orig_serange:
                    if not module.check_mode:
                        sefcontext.modify(target, setype, ftype, serange, seuser)
                    changed = True
                    if module._diff:
                        prepared_diff += '# Change to semanage file context mappings\n'
                        prepared_diff += '-%s      %s      %s:%s:%s:%s\n' % (target, ftype, orig_seuser, orig_serole, orig_setype, orig_serange)
                        prepared_diff += '+%s      %s      %s:%s:%s:%s\n' % (target, ftype, seuser, orig_serole, setype, serange)
            else:
                if seuser is None:
                    seuser = 'system_u'
                if serange is None:
                    serange = 's0'
                if not module.check_mode:
                    sefcontext.add(target, setype, ftype, serange, seuser)
                changed = True
                if module._diff:
                    prepared_diff += '# Addition to semanage file context mappings\n'
                    prepared_diff += '+%s      %s      %s:%s:%s:%s\n' % (target, ftype, seuser, 'object_r', setype, serange)
        else:
            exists = semanage_fcontext_substitute_exists(sefcontext, target)
            if exists:
                orig_substitute = exists
                if substitute != orig_substitute:
                    if not module.check_mode:
                        sefcontext.modify_equal(target, substitute)
                    changed = True
                    if module._diff:
                        prepared_diff += '# Change to semanage file context path substitutions\n'
                        prepared_diff += '-%s = %s\n' % (target, orig_substitute)
                        prepared_diff += '+%s = %s\n' % (target, substitute)
            else:
                if not module.check_mode:
                    sefcontext.add_equal(target, substitute)
                changed = True
                if module._diff:
                    prepared_diff += '# Addition to semanage file context path substitutions\n'
                    prepared_diff += '+%s = %s\n' % (target, substitute)
    except Exception as e:
        module.fail_json(msg='%s: %s\n' % (e.__class__.__name__, to_native(e)))
    if module._diff and prepared_diff:
        result['diff'] = dict(prepared=prepared_diff)
    module.exit_json(changed=changed, seuser=seuser, serange=serange, **result)