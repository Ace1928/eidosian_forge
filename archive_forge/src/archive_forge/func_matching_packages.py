from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import shlex_quote
def matching_packages(module, name):
    ports_glob_path = module.get_bin_path('ports_glob', True)
    rc, out, err = module.run_command('%s %s' % (ports_glob_path, name))
    occurrences = out.count('\n')
    if occurrences == 0:
        name_without_digits = re.sub('[0-9]', '', name)
        if name != name_without_digits:
            rc, out, err = module.run_command('%s %s' % (ports_glob_path, name_without_digits))
            occurrences = out.count('\n')
    return occurrences