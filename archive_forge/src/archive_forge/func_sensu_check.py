from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def sensu_check(module, path, name, state='present', backup=False):
    changed = False
    reasons = []
    stream = None
    try:
        try:
            stream = open(path, 'r')
            config = json.load(stream)
        except IOError as e:
            if e.errno == 2:
                if state == 'absent':
                    reasons.append("file did not exist and state is `absent'")
                    return (changed, reasons)
                config = {}
            else:
                module.fail_json(msg=to_native(e), exception=traceback.format_exc())
        except ValueError:
            msg = '{path} contains invalid JSON'.format(path=path)
            module.fail_json(msg=msg)
    finally:
        if stream:
            stream.close()
    if 'checks' not in config:
        if state == 'absent':
            reasons.append("`checks' section did not exist and state is `absent'")
            return (changed, reasons)
        config['checks'] = {}
        changed = True
        reasons.append("`checks' section did not exist")
    if state == 'absent':
        if name in config['checks']:
            del config['checks'][name]
            changed = True
            reasons.append("check was present and state is `absent'")
    if state == 'present':
        if name not in config['checks']:
            check = {}
            config['checks'][name] = check
            changed = True
            reasons.append("check was absent and state is `present'")
        else:
            check = config['checks'][name]
        simple_opts = ['command', 'handlers', 'subscribers', 'interval', 'timeout', 'ttl', 'handle', 'dependencies', 'standalone', 'publish', 'occurrences', 'refresh', 'aggregate', 'low_flap_threshold', 'high_flap_threshold', 'source']
        for opt in simple_opts:
            if module.params[opt] is not None:
                if opt not in check or check[opt] != module.params[opt]:
                    check[opt] = module.params[opt]
                    changed = True
                    reasons.append("`{opt}' did not exist or was different".format(opt=opt))
            elif opt in check:
                del check[opt]
                changed = True
                reasons.append("`{opt}' was removed".format(opt=opt))
        if module.params['custom']:
            custom_params = module.params['custom']
            overwrited_fields = set(custom_params.keys()) & set(simple_opts + ['type', 'subdue', 'subdue_begin', 'subdue_end'])
            if overwrited_fields:
                msg = 'You can\'t overwriting standard module parameters via "custom". You are trying overwrite: {opt}'.format(opt=list(overwrited_fields))
                module.fail_json(msg=msg)
            for k, v in custom_params.items():
                if k in config['checks'][name]:
                    if not config['checks'][name][k] == v:
                        changed = True
                        reasons.append("`custom param {opt}' was changed".format(opt=k))
                else:
                    changed = True
                    reasons.append("`custom param {opt}' was added".format(opt=k))
                check[k] = v
            simple_opts += custom_params.keys()
        for opt in set(config['checks'][name].keys()) - set(simple_opts + ['type', 'subdue', 'subdue_begin', 'subdue_end']):
            changed = True
            reasons.append("`custom param {opt}' was deleted".format(opt=opt))
            del check[opt]
        if module.params['metric']:
            if 'type' not in check or check['type'] != 'metric':
                check['type'] = 'metric'
                changed = True
                reasons.append("`type' was not defined or not `metric'")
        if not module.params['metric'] and 'type' in check:
            del check['type']
            changed = True
            reasons.append("`type' was defined")
        if module.params['subdue_begin'] is not None and module.params['subdue_end'] is not None:
            subdue = {'begin': module.params['subdue_begin'], 'end': module.params['subdue_end']}
            if 'subdue' not in check or check['subdue'] != subdue:
                check['subdue'] = subdue
                changed = True
                reasons.append("`subdue' did not exist or was different")
        elif 'subdue' in check:
            del check['subdue']
            changed = True
            reasons.append("`subdue' was removed")
    if changed and (not module.check_mode):
        if backup:
            module.backup_local(path)
        try:
            try:
                stream = open(path, 'w')
                stream.write(json.dumps(config, indent=2) + '\n')
            except IOError as e:
                module.fail_json(msg=to_native(e), exception=traceback.format_exc())
        finally:
            if stream:
                stream.close()
    return (changed, reasons)