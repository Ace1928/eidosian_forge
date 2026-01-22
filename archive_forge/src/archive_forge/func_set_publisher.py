from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def set_publisher(module, params):
    name = params['name']
    args = []
    if params['origin'] is not None:
        args.append('--remove-origin=*')
        args.extend(['--add-origin=' + u for u in params['origin']])
    if params['mirror'] is not None:
        args.append('--remove-mirror=*')
        args.extend(['--add-mirror=' + u for u in params['mirror']])
    if params['sticky'] is not None and params['sticky']:
        args.append('--sticky')
    elif params['sticky'] is not None:
        args.append('--non-sticky')
    if params['enabled'] is not None and params['enabled']:
        args.append('--enable')
    elif params['enabled'] is not None:
        args.append('--disable')
    rc, out, err = module.run_command(['pkg', 'set-publisher'] + args + [name], check_rc=True)
    response = {'rc': rc, 'results': [out], 'msg': err, 'changed': True}
    if rc != 0:
        module.fail_json(**response)
    module.exit_json(**response)