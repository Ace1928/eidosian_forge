from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def query_log_status(module, le_path, path, state='present'):
    """ Returns whether a log is followed or not. """
    if state == 'present':
        rc, out, err = module.run_command([le_path, 'followed', path])
        if rc == 0:
            return True
        return False