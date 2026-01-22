from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def selfupdate(module, port_path):
    """ Update Macports and the ports tree. """
    rc, out, err = module.run_command('%s -v selfupdate' % port_path)
    if rc == 0:
        updated = any((re.search('Total number of ports parsed:\\s+[^0]', s.strip()) or re.search('Installing new Macports release', s.strip()) for s in out.split('\n') if s))
        if updated:
            changed = True
            msg = 'Macports updated successfully'
        else:
            changed = False
            msg = 'Macports already up-to-date'
        return (changed, msg, out, err)
    else:
        module.fail_json(msg='Failed to update Macports', stdout=out, stderr=err)