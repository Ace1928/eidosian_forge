from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule  # noqa: E402
def runlabel(module, executable):
    changed = False
    command = [executable, 'container', 'runlabel']
    command.append(module.params['label'])
    command.append(module.params['image'])
    rc, out, err = module.run_command(command)
    if rc == 0:
        changed = True
    else:
        module.fail_json(msg='Error running the runlabel from image %s: %s' % (module.params['image'], err))
    return (changed, out, err)