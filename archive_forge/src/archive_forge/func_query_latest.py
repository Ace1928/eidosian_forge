from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def query_latest(module, name):
    cmd = '%s version %s' % (APK_PATH, name)
    rc, stdout, stderr = module.run_command(cmd, check_rc=False)
    search_pattern = '(%s)-[\\d\\.\\w]+-[\\d\\w]+\\s+(.)\\s+[\\d\\.\\w]+-[\\d\\w]+\\s+' % re.escape(name)
    match = re.search(search_pattern, stdout)
    if match and match.group(2) == '<':
        return False
    return True